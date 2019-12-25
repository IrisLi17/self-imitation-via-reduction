from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np


class ParallelSubprocVecEnv(SubprocVecEnv):
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        # TODO: dispatch method_kwargs. Now method_kwargs is the same for every remote
        dispatched_args = [[] for _ in range(len(target_remotes))]
        for args in method_args:
            assert isinstance(args, list) or isinstance(args, tuple)
            for i in range(len(target_remotes)):
                dispatched_args[i].append(args[i])
        for i, remote in enumerate(target_remotes):
            remote.send(('env_method', (method_name, dispatched_args[i], method_kwargs)))
        return [remote.recv() for remote in target_remotes]