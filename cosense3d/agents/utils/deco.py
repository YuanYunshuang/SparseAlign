
from cosense3d.agents.core.hooks import CheckPointsHook


def save_ckpt_on_error(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            CheckPointsHook.save(args[0], f'debug_ep{args[0].epoch - 1}.pth')
            print(f"Exception caught in {func.__name__}: {e}")
            raise e
    return wrapper


