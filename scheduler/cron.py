from apscheduler.schedulers.background import BackgroundScheduler
from model.train import train_model

scheduler = BackgroundScheduler()

@scheduler.scheduled_job("cron", hour=5)
def scheduled_job():
    print("⏰ 執行每日自動訓練...")
    train_model("2330")

def start_scheduler():
    scheduler.start()
