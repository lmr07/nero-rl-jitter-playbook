"""Exercise the actual task's action-to-command path without Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
from test_nero_actuator import model

tree = ast.parse((Path(__file__).resolve().parents[1]/'envs/nero/nero_fixed.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeroFixed')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
# Execute the production prefix through host-command adaptation, excluding physics.
stop = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Expr) and
            isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'render')
method.body = method.body[:stop] + [ast.Return(value=ast.Name(id='target', ctx=ast.Load()))]
module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
ns = {'torch':torch};exec(compile(module, '<Nero step command prefix>', 'exec'), ns)

class IncrementTests(unittest.TestCase):
    def test_accumulates_sent_target_and_zero_holds(self):
        a=model();a.host_elapsed.fill_(3.)
        env=NS(device='cpu', num_envs=2, actuator=a, incremental=True,
            cfg=NS(normalization=NS(clip_actions=100),control=NS(action_scale=.25,target_delta_limit=.012)),
            default_dof_pos=torch.zeros(9), lower=torch.full((9,),-3.), upper=torch.full((9,),3.),
            actions=torch.zeros(2,7),last_actions=torch.zeros(2,7),candidate_actions=torch.zeros(2,7))
        for i in range(5):
            result=ns['step'](env,torch.full((2,7),100.))
            torch.testing.assert_close(result[:,:7],torch.full((2,7),.012*(i+1)))
        result=ns['step'](env,torch.zeros(2,7))
        torch.testing.assert_close(result[:,:7],torch.full((2,7),.060))
        torch.testing.assert_close(env.actions,result[:,:7]/.25)
        result=ns['step'](env,torch.full((2,7),-100.))
        torch.testing.assert_close(result[:,:7],torch.full((2,7),.048))
        env.incremental=False
        result=ns['step'](env,torch.zeros(2,7))
        # Legacy absolute target of zero still uses the original .02 slew limit.
        torch.testing.assert_close(result[:,:7],torch.full((2,7),.028))

if __name__=='__main__':unittest.main()
