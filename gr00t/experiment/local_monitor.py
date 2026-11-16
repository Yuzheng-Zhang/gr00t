"""Rank-zero progress and loss artifacts, without an online tracking service."""
import html
import json
import logging
import math
from pathlib import Path
import time

from transformers import TrainerCallback


class LocalTrainingMonitor(TrainerCallback):
    def __init__(self):
        self.points = []
        self.phase = 'training'
        self.last_checkpoint_step = None

    def _write(self, args, state):
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        record = {
            'phase': self.phase, 'step': state.global_step, 'max_steps': state.max_steps,
            'latest_loss': self.points[-1]['loss'] if self.points else None,
            'loss_step': self.points[-1]['step'] if self.points else None,
            'last_checkpoint_step': self.last_checkpoint_step,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        def atomic(name, text):
            target = root / name
            temp = target.with_suffix(target.suffix + '.tmp')
            temp.write_text(text, encoding='utf-8')
            temp.replace(target)
        atomic('training_status.json', json.dumps(record, indent=2, ensure_ascii=False))
        labels = {'training': 'Training', 'saving_final': 'Steps finished; saving final model',
                  'completed': 'Training and final save completed'}
        plot = '<img src="loss.png" style="width:100%;max-width:1000px">' if self.points else '<p>Waiting for the first logged loss.</p>'
        atomic('progress.html', f'''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10"><title>GR00T training</title></head>
<body style="font-family:system-ui;max-width:1000px;margin:40px auto;padding:20px">
<h1>GR00T training</h1><p>{labels[self.phase]}</p>
<h2>Step {state.global_step} / {state.max_steps}</h2>
<progress style="width:100%" value="{state.global_step}" max="{max(state.max_steps,1)}"></progress>
<p>Latest logged loss: {record['latest_loss']} (step {record['loss_step']})</p>
<p>Last checkpoint: {self.last_checkpoint_step}; updated: {html.escape(record['updated_at'])}</p>
{plot}<p>This page refreshes every 10 seconds. If the timestamp stops changing, inspect train.log.
A decreasing training loss does not establish real-robot task success.</p></body></html>''')

    def _safe_update(self, args, state, plot=False):
        if not state.is_world_process_zero:
            return
        try:
            if plot and self.points:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot([p['step'] for p in self.points], [p['loss'] for p in self.points],
                        linewidth=1.3, marker='.' if len(self.points)<30 else None)
                ax.set(xlabel='Optimizer step', ylabel='Logged training loss', title='GR00T training loss')
                ax.grid(alpha=.25)
                fig.tight_layout()
                root=Path(args.output_dir)
                root.mkdir(parents=True,exist_ok=True)
                temporary=root/'loss.tmp.png'
                try:
                    fig.savefig(temporary,dpi=130)
                    temporary.replace(root/'loss.png')
                finally:
                    plt.close(fig)
            self._write(args,state)
        except Exception:
            logging.exception('Could not update local training monitor; training continues.')

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.points = [{'step': int(p['step']), 'loss': float(p['loss'])}
                       for p in state.log_history if 'loss' in p and 'step' in p
                       and math.isfinite(float(p['loss']))]
        self._safe_update(args,state,plot=True)

    def on_step_end(self, args, state, control, **kwargs):
        self._safe_update(args,state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs or 'loss' not in logs:
            return
        value = float(logs['loss'])
        if not math.isfinite(value):
            logging.error('Nonfinite logged loss at step %s: %s',state.global_step,value)
            return
        row={'step':state.global_step,'loss':value}
        for key in ['learning_rate','grad_norm','epoch']:
            if key in logs:
                row[key]=float(logs[key])
        self.points.append(row)
        try:
            root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
            with (root/'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        except OSError:
            logging.exception('Could not append metrics; training continues.')
        print(f'[progress] step {state.global_step}/{state.max_steps} | loss={value:.6g}',flush=True)
        self._safe_update(args,state,plot=True)

    def on_save(self,args,state,control,**kwargs):
        if state.is_world_process_zero:
            self.last_checkpoint_step=state.global_step
            self._safe_update(args,state)

    def on_train_end(self,args,state,control,**kwargs):
        self.phase='saving_final'
        self._safe_update(args,state)

    def mark_completed(self,args,state):
        self.phase='completed'
        self._safe_update(args,state)
