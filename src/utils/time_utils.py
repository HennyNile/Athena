from datetime import datetime

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]