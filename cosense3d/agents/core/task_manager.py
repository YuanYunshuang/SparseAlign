
from collections import OrderedDict


class TaskManager:
    def __init__(self):
        pass

    def summarize_tasks(self, tasks):
        tasks_out = {0: {'no_grad': [], 'with_grad': []},
                     1: {'no_grad': [], 'with_grad': []},
                     2: {'no_grad': [], 'with_grad': []},
                     3: {'loss': []}}
        no_grad0, no_grad1, no_grad2, _ = self.reformat_tasks(tasks['no_grad'])
        with_grad0, with_grad1, with_grad2, _ = self.reformat_tasks(tasks['with_grad'])
        tasks_out[0]['no_grad'] = no_grad0
        tasks_out[0]['with_grad'] = with_grad0
        tasks_out[1]['no_grad'] = no_grad1
        tasks_out[1]['with_grad'] = with_grad1
        tasks_out[2]['no_grad'] = no_grad2
        tasks_out[2]['with_grad'] = with_grad2
        tasks_out[3]['loss'] = self.reformat_tasks(tasks['loss'])[3]
        return tasks_out

    def summarize_loss_tasks(self, tasks):
        return self.reformat_tasks(tasks)

    def reformat_tasks(self, task_list):
        task_out = ({}, {}, {}, {})  # two stages
        if len(task_list) == 0:
            return task_out
        for task in task_list:
            cav_id, task_label, args = task
            stage_order, task_name = task_label.split(':')
            stage = int(stage_order[0])
            order = int(stage_order[1:])
            task_name = task_name.strip()
            if order not in task_out[stage]:
                task_out[stage][order] = {}
            if task_name not in task_out[stage][order]:
                task_out[stage][order][task_name] = []
            task_out[stage][order][task_name].append((cav_id, args))

        task_out = [self.task_to_ordered_dict(tasks) for tasks in task_out]
        return task_out

    def task_to_ordered_dict(self, tasks):
        orders = sorted(tasks)
        ordered_task = OrderedDict()
        for i in orders:
            for k, v in tasks[i].items():
                ordered_task[k] = v
        return ordered_task





