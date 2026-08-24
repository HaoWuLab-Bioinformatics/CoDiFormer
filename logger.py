import os
import torch


class Logger:
    """Simple training logger for final experiment results."""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.test = None

    def add_result(self, run, result):
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def _select_epoch(self, result, mode='max_acc'):
        if mode == 'max_acc':
            return result[:, 1].argmax().item()
        return result[:, 3].argmin().item()

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run], dtype=torch.float)
            idx = self._select_epoch(result, mode)

            print(f'Run {run + 1:02d}:')
            print(f'Best Train: {result[:,0].max():.2f}')
            print(f'Best Valid: {result[:,1].max():.2f}')
            print(f'Best Test: {result[:,2].max():.2f}')
            print(f'Final Test: {result[idx,2]:.2f}')

            self.test = result[idx, 2]
            return self.test

        results = [
            100 * torch.tensor(r, dtype=torch.float)
            for r in self.results if len(r) > 0
        ]
        if not results:
            print('No valid results.')
            return torch.tensor([])

        best = []
        for r in results:
            idx = self._select_epoch(r, mode)
            best.append(r[idx, 2])

        best = torch.stack(best)
        print(f'All runs Test: {best.mean():.2f} ± {best.std():.2f}')
        self.test = best.mean()
        return best


def save_result(args, results):
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', f'{args.method}_result.txt')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(str(results) + '\n')


def get_process_memory_mb():
    return float('nan')
