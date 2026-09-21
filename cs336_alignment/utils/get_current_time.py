from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time() -> str:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return now.strftime("%Y%m%d_%H%M%S")