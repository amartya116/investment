from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from models import Base, RawStockData, Company, StreakStatistic, LongStreak
import pandas as pd
from datetime import datetime
from db_config import DBConfig
import numpy as np

class DatabaseManager:
    def __init__(self, config: DBConfig):
        self.engine = create_engine(
            "postgresql+psycopg2://postgres:amartya@localhost:5432/stock_analysis"
        )
        self.Session = sessionmaker(bind=self.engine)

    def _convert_numpy_types(self, data: dict) -> dict:
        """Convert NumPy data types to native Python types"""
        return {
            k: (
                float(v) if isinstance(v, np.floating)
                else int(v) if isinstance(v, np.integer)
                else v
            )
            for k, v in data.items()
        }

    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(self.engine)

    def save_raw_stock_data(self, ticker: str, df: pd.DataFrame):
        """Save raw stock data to database"""
        session = self.Session()
        try:
            records = []
            for date, row in df.iterrows():
                record = {
                    'ticker': ticker,
                    'date': date,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                }
                records.append(record)

            stmt = insert(RawStockData).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint='uix_ticker_date',
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume
                }
            )

            session.execute(stmt)
            session.commit()
        finally:
            session.close()

    def get_raw_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve raw stock data from database"""
        session = self.Session()
        try:
            query = session.query(
                RawStockData.date,
                RawStockData.open,
                RawStockData.high,
                RawStockData.low,
                RawStockData.close,
                RawStockData.volume
            ).filter(
                RawStockData.ticker == ticker,
                RawStockData.date.between(start_date, end_date)
            ).order_by(RawStockData.date)

            df = pd.read_sql(query.statement, session.bind, index_col='date')
            return df
        finally:
            session.close()

    def save_streak_statistics(self, ticker: str, analysis_date: datetime, timeframe_months: int, stats: dict):
        """Save streak statistics to database"""
        session = self.Session()
        try:
            clean_stats = self._convert_numpy_types(stats)

            stmt = insert(StreakStatistic).values(
                ticker=ticker,
                analysis_date=analysis_date,
                timeframe_months=timeframe_months,
                max_up_streak=clean_stats['max_up_streak'],
                max_down_streak=clean_stats['max_down_streak'],
                max_up_change=clean_stats['max_up_change'],
                max_down_change=clean_stats['max_down_change'],
                max_up_change_pct=clean_stats['max_up_change_pct'],
                max_down_change_pct=clean_stats['max_down_change_pct'],
                avg_up_change=clean_stats['avg_up_change'],
                avg_down_change=clean_stats['avg_down_change'],
                avg_up_change_pct=clean_stats['avg_up_change_pct'],
                avg_down_change_pct=clean_stats['avg_down_change_pct']
            )

            stmt = stmt.on_conflict_do_update(
                constraint='streak_statistics_ticker_analysis_date_timeframe_months_key',
                set_={
                    'max_up_streak': stmt.excluded.max_up_streak,
                    'max_down_streak': stmt.excluded.max_down_streak,
                    'max_up_change': stmt.excluded.max_up_change,
                    'max_down_change': stmt.excluded.max_down_change,
                    'max_up_change_pct': stmt.excluded.max_up_change_pct,
                    'max_down_change_pct': stmt.excluded.max_down_change_pct,
                    'avg_up_change': stmt.excluded.avg_up_change,
                    'avg_down_change': stmt.excluded.avg_down_change,
                    'avg_up_change_pct': stmt.excluded.avg_up_change_pct,
                    'avg_down_change_pct': stmt.excluded.avg_down_change_pct
                }
            )

            session.execute(stmt)
            session.commit()
        finally:
            session.close()

    def save_long_streaks(self, ticker: str, streaks: list):
        """Save long streak information to database"""
        session = self.Session()
        try:
            records = []
            for streak in streaks:
                record = self._convert_numpy_types({
                    'ticker': ticker,
                    'streak_type': streak['type'],
                    'start_date': streak['start_date'],
                    'end_date': streak['end_date'],
                    'length': streak['length'],
                    'total_change': streak['change'],
                    'total_change_pct': streak['change_pct'],
                    'next_day_change': streak['next_day_change'],
                    'next_day_change_pct': streak['next_day_change_pct']
                })
                records.append(record)

            stmt = insert(LongStreak).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint='long_streaks_ticker_start_date_streak_type_key',
                set_={
                    'end_date': stmt.excluded.end_date,
                    'length': stmt.excluded.length,
                    'total_change': stmt.excluded.total_change,
                    'total_change_pct': stmt.excluded.total_change_pct,
                    'next_day_change': stmt.excluded.next_day_change,
                    'next_day_change_pct': stmt.excluded.next_day_change_pct
                }
            )

            session.execute(stmt)
            session.commit()
        finally:
            session.close()

    def save_company_info(self, ticker: str, company_data: dict):
        """Save or update company information"""
        session = self.Session()
        try:
            clean_data = self._convert_numpy_types(company_data)

            stmt = insert(Company).values(
                ticker=ticker,
                name=clean_data.get('name'),
                exchange=clean_data.get('exchange'),
                sector=clean_data.get('sector'),
                industry=clean_data.get('industry'),
                market_cap=clean_data.get('market_cap'),
                last_updated=datetime.utcnow()
            )

            stmt = stmt.on_conflict_do_update(
                constraint='companies_ticker_key',
                set_={
                    'name': stmt.excluded.name,
                    'exchange': stmt.excluded.exchange,
                    'sector': stmt.excluded.sector,
                    'industry': stmt.excluded.industry,
                    'market_cap': stmt.excluded.market_cap,
                    'last_updated': stmt.excluded.last_updated
                }
            )

            session.execute(stmt)
            session.commit()
        finally:
            session.close()
