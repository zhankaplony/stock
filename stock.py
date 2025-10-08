# -*- coding: utf-8 -*-
"""
股价预测模型 - Streamlit应用版 (集成学习优化版 + Numba加速)
基于突破均线模式的稳健预测模型，支持集成学习和智能缓存
使用Numba加速技术指标计算 - 修复版本 + 前复权数据
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import akshare as ak
import warnings
import os
import matplotlib as mpl
from matplotlib import font_manager
import pickle
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import gc

# 尝试导入numba，如果不可用则使用备选方案
try:
    from numba import jit, njit, prange, float64, int64
    from numba.types import Array

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# 常量定义
CACHE_DIR = "./stock_cache"
CACHE_CONFIG_FILE = os.path.join(CACHE_DIR, "cache_config.pkl")
DEFAULT_START_YEARS = 3
DEFAULT_TEST_DAYS = 90
MAX_CACHE_AGE_HOURS = 24
REPLAY_CACHE_AGE_HOURS = 24 * 7


class Config:
    """配置类"""
    # 优化的单个模型参数（为集成学习调整）
    OPTIMIZED_MODEL_PARAMS = {
        'RandomForest': {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'random_state': 42,
            'n_jobs': -1
        },
        'XGBoost': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        },
        'LightGBM': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        },
        'GradientBoosting': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
    }

    # 集成模型配置
    ENSEMBLE_CONFIG = {
        'voting': {
            'voting': 'soft',  # 使用概率投票
            'weights': [2, 3, 2]  # 模型权重 [RF, XGB, LGB]
        },
        'stacking': {
            'final_estimator': LogisticRegression(max_iter=1000, random_state=42),
            'cv': 3
        }
    }

    FEATURE_WINDOWS = [5, 10, 20, 30, 60]
    RETURN_PERIODS = [1, 2, 3, 5, 10]


# 创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)


def setup_chinese_font() -> bool:
    """设置中文字体"""
    try:
        # 字体路径列表
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/simkai.ttf',
            'C:/Windows/Fonts/simsun.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]

        font_path = next((path for path in font_paths if os.path.exists(path)), None)

        if font_path:
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = [font_prop.get_name(), 'DejaVu Sans', 'sans-serif']
        else:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

        # 统一设置样式
        plt.style.use('dark_background')
        plt.rcParams.update({
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'legend.facecolor': 'black',
            'legend.edgecolor': 'white',
            'legend.labelcolor': 'white',
            'figure.facecolor': 'black',
            'axes.facecolor': 'black',
            'savefig.facecolor': 'black',
            'axes.unicode_minus': False,
            'figure.titlesize': 14,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })

        sns.set_style("darkgrid")
        sns.set_palette("deep")
        return True

    except Exception as e:
        logger.warning(f"字体设置失败: {e}")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        return False


# 初始化字体
setup_chinese_font()


class NumbaAccelerator:
    """Numba加速器类 - 用于加速技术指标计算（修复版本）"""

    @staticmethod
    @njit(float64[:](float64[:], int64), cache=True, parallel=False)
    def rolling_mean_numba(arr, window):
        """使用Numba加速的滚动均值计算"""
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(window - 1, n):
            total = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]):
                    total += arr[j]
                    count += 1
            if count > 0:
                result[i] = total / count
        return result

    # 修复rolling_std_numba函数中的计算错误
    @staticmethod
    @njit(float64[:](float64[:], int64), cache=True, parallel=False)
    def rolling_std_numba(arr, window):
        """使用Numba加速的滚动标准差计算 - 修复版本"""
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(window - 1, n):
            # 收集有效值并计算均值
            total = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]):
                    total += arr[j]
                    count += 1

            if count < 2:  # 至少需要2个点计算标准差
                continue

            mean_val = total / count

            # 计算方差
            variance = 0.0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]):
                    variance += (arr[j] - mean_val) ** 2
            variance /= (count - 1)  # 样本方差

            result[i] = np.sqrt(variance)
        return result

    @staticmethod
    @njit(float64[:](float64[:], int64), cache=True, parallel=False)
    def rolling_min_numba(arr, window):
        """使用Numba加速的滚动最小值计算"""
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(window - 1, n):
            min_val = np.inf
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]) and arr[j] < min_val:
                    min_val = arr[j]
            if min_val != np.inf:
                result[i] = min_val
        return result

    @staticmethod
    @njit(float64[:](float64[:], int64), cache=True, parallel=False)
    def rolling_max_numba(arr, window):
        """使用Numba加速的滚动最大值计算"""
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)

        for i in range(window - 1, n):
            max_val = -np.inf
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]) and arr[j] > max_val:
                    max_val = arr[j]
            if max_val != -np.inf:
                result[i] = max_val
        return result

    @staticmethod
    def accelerate_dataframe_operations(df, use_numba=True):
        """加速DataFrame操作"""
        if not use_numba or not NUMBA_AVAILABLE:
            return df

        try:
            # 转换列为numpy数组进行加速计算
            close_prices = df['收盘'].values.astype(np.float64)
            high_prices = df['最高'].values.astype(np.float64)
            low_prices = df['最低'].values.astype(np.float64)
            volume = df['成交量'].values.astype(np.float64)

            # 计算技术指标
            for window in Config.FEATURE_WINDOWS:
                # 移动平均线
                ma_key = f'ma_{window}'
                df[ma_key] = NumbaAccelerator.rolling_mean_numba(close_prices, window)

                # 支撑阻力
                support_key = f'support_{window}'
                resistance_key = f'resistance_{window}'
                df[support_key] = NumbaAccelerator.rolling_min_numba(low_prices, window)
                df[resistance_key] = NumbaAccelerator.rolling_max_numba(high_prices, window)

            # 计算波动率
            returns = np.zeros(len(close_prices))
            for i in range(1, len(close_prices)):
                if close_prices[i - 1] != 0 and not np.isnan(close_prices[i - 1]):
                    returns[i] = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]
                else:
                    returns[i] = np.nan

            for window in [5, 20]:
                vol_key = f'volatility_{window}'
                df[vol_key] = NumbaAccelerator.rolling_std_numba(returns, window)

        except Exception as e:
            logger.warning(f"Numba加速失败，回退到pandas计算: {e}")

        return df


class CacheManager:
    """缓存管理类"""

    @staticmethod
    def get_cache_key(symbol: str, start_date: str, end_date: str = None,
                      analysis_mode: str = "replay") -> str:
        """生成缓存键"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        key_data = f"{symbol}_{start_date}_{end_date}_{analysis_mode}"
        return hashlib.md5(key_data.encode()).hexdigest()

    @staticmethod
    def load_config() -> Dict:
        """加载缓存配置"""
        try:
            if os.path.exists(CACHE_CONFIG_FILE):
                with open(CACHE_CONFIG_FILE, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"加载缓存配置失败: {e}")
        return {}

    @staticmethod
    def save_config(config: Dict) -> bool:
        """保存缓存配置"""
        try:
            with open(CACHE_CONFIG_FILE, 'wb') as f:
                pickle.dump(config, f)
            return True
        except Exception as e:
            logger.warning(f"保存缓存配置失败: {e}")
            return False

    @staticmethod
    def get_last_start_date() -> Optional[str]:
        """获取上次使用的开始日期"""
        config = CacheManager.load_config()
        return config.get('last_start_date')

    @staticmethod
    def save_last_start_date(start_date: str) -> bool:
        """保存当前使用的开始日期"""
        config = CacheManager.load_config()
        config['last_start_date'] = start_date
        return CacheManager.save_config(config)

    @staticmethod
    def save_to_cache(symbol: str, start_date: str, data: pd.DataFrame,
                      end_date: str = None, analysis_mode: str = "replay") -> bool:
        """保存数据到缓存"""
        try:
            cache_key = CacheManager.get_cache_key(symbol, start_date, end_date, analysis_mode)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

            cache_data = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'analysis_mode': analysis_mode,
                'data': data,
                'timestamp': datetime.now()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            return True
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            return False

    @staticmethod
    def load_from_cache(symbol: str, start_date: str, end_date: str = None,
                        analysis_mode: str = "replay", max_age_hours: int = None) -> Optional[pd.DataFrame]:
        """从缓存加载数据 - 修复文件处理"""
        try:
            cache_key = CacheManager.get_cache_key(symbol, start_date, end_date, analysis_mode)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

            if not os.path.exists(cache_file):
                return None

            # 检查文件大小，避免损坏文件
            if os.path.getsize(cache_file) == 0:
                os.remove(cache_file)  # 删除损坏的缓存文件
                return None

            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # 设置缓存过期时间
            if max_age_hours is None:
                max_age_hours = REPLAY_CACHE_AGE_HOURS if analysis_mode == "replay" else MAX_CACHE_AGE_HOURS

            cache_age = datetime.now() - cache_data['timestamp']
            if cache_age.total_seconds() > max_age_hours * 3600:
                os.remove(cache_file)  # 删除过期缓存
                return None

            return cache_data['data']

        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            logger.warning(f"缓存文件损坏: {e}")
            try:
                os.remove(cache_file)  # 删除损坏文件
            except:
                pass
            return None
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None

    @staticmethod
    def clear_cache() -> bool:
        """清空缓存"""
        try:
            for file in os.listdir(CACHE_DIR):
                if file.endswith('.pkl') and file != "cache_config.pkl":
                    os.remove(os.path.join(CACHE_DIR, file))
            return True
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    @staticmethod
    def get_cache_info() -> Dict:
        """获取缓存信息"""
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl') and f != "cache_config.pkl"]
        return {
            'file_count': len(cache_files),
            'total_size': sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
        }


class StockDataFetcher:
    """股票数据获取器 - 修复版本（使用前复权）"""

    @staticmethod
    def get_stock_name(symbol: str) -> str:
        """获取股票名称"""
        try:
            # 方法1: 从A股代码名称映射获取
            try:
                stock_info = ak.stock_info_a_code_name()
                if not stock_info.empty:
                    match = stock_info[stock_info['code'] == symbol]
                    if not match.empty:
                        return match.iloc[0]['name']
            except:
                pass

            # 方法2: 从个股信息获取
            try:
                stock_individual_info = ak.stock_individual_info_em(symbol=symbol)
                if not stock_individual_info.empty and 'value' in stock_individual_info.columns:
                    for idx, row in stock_individual_info.iterrows():
                        if '简称' in str(row.get('item', '')) or '名称' in str(row.get('item', '')):
                            return row['value']
            except:
                pass

            return f"股票{symbol}"

        except Exception as e:
            logger.warning(f"获取股票名称失败 {symbol}: {e}")
            return f"股票{symbol}"

    @staticmethod
    def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取股票数据 - 使用前复权数据"""
        try:
            # 方法1: 使用akshare的前复权数据
            try:
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"  # 关键修复：使用前复权
                )

                if stock_data.empty:
                    st.warning(f"股票 {symbol} 无前复权数据返回，尝试获取不复权数据")
                    # 备用方案：获取不复权数据
                    stock_data = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date
                    )
            except Exception as e:
                logger.warning(f"前复权数据获取失败，尝试不复权数据: {e}")
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date
                )

            if stock_data.empty:
                st.warning(f"股票 {symbol} 无数据返回")
                return None

            # 修复列名可能的变化
            column_mapping = {
                '日期': 'date', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            }

            stock_data.columns = [column_mapping.get(col, col) for col in stock_data.columns]

            # 确保必要列存在
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]

            if missing_cols:
                st.error(f"缺少必要列: {missing_cols}")
                return None

            # 数据预处理
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)

            # 重命名列为中文（与后续代码兼容）
            stock_data = stock_data.rename(columns={
                'open': '开盘', 'high': '最高', 'low': '最低',
                'close': '收盘', 'volume': '成交量'
            })

            # 验证数据质量 - 检查是否有价格断崖
            price_changes = stock_data['收盘'].pct_change().abs()
            large_gaps = price_changes[price_changes > 0.3]  # 超过30%的价格变动可能有问题

            if len(large_gaps) > 0:
                st.warning(f"检测到 {len(large_gaps)} 个可能的价格断崖，建议检查数据质量")
                logger.warning(f"股票 {symbol} 检测到价格断崖: {large_gaps.index.tolist()}")

            return stock_data[['开盘', '最高', '最低', '收盘', '成交量']]

        except Exception as e:
            st.error(f"获取股票数据失败: {e}")
            return None

    @staticmethod
    def fetch_stock_data_enhanced(symbol: str, start_date: str, end_date: str, max_retries: int = 3) -> Optional[
        pd.DataFrame]:
        """增强的股票数据获取 - 多数据源尝试"""
        methods = [
            # 方法1: akshare前复权
            lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                       start_date=start_date, end_date=end_date, adjust="qfq"),
            # 方法2: akshare不复权（备用）
            lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                       start_date=start_date, end_date=end_date),
        ]

        for i, method in enumerate(methods):
            try:
                stock_data = method()
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"成功获取 {symbol} 数据 (方法{i + 1})")

                    # 数据格式处理
                    column_mapping = {
                        '日期': 'date', '开盘': 'open', '最高': 'high',
                        '最低': 'low', '收盘': 'close', '成交量': 'volume'
                    }
                    stock_data.columns = [column_mapping.get(col, col) for col in stock_data.columns]

                    # 确保必要列存在
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in stock_data.columns for col in required_cols):
                        stock_data['date'] = pd.to_datetime(stock_data['date'])
                        stock_data.set_index('date', inplace=True)
                        stock_data = stock_data.rename(columns={
                            'open': '开盘', 'high': '最高', 'low': '最低',
                            'close': '收盘', 'volume': '成交量'
                        })
                        return stock_data[['开盘', '最高', '最低', '收盘', '成交量']]

            except Exception as e:
                logger.warning(f"获取 {symbol} 数据方法{i + 1}失败: {e}")
                continue

        st.error(f"所有数据获取方法都失败: {symbol}")
        return None

    @staticmethod
    def generate_simulated_data(start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
        """生成模拟数据"""
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        dates = dates[dates.dayofweek < 5]  # 只保留工作日

        np.random.seed(seed)
        n = len(dates)

        # 创建更真实的价格序列
        trend = np.linspace(0, 0.15, n)
        seasonal = 0.08 * np.sin(2 * np.pi * np.arange(n) / 30)
        noise = np.random.normal(0, 0.012, n)

        log_prices = trend + seasonal + np.cumsum(noise)
        base_prices = 100 * np.exp(log_prices)

        # 生成OHLC数据
        df = pd.DataFrame({
            '开盘': base_prices * (1 + np.random.normal(0, 0.002, n)),
            '最高': base_prices * (1 + np.abs(np.random.normal(0, 0.006, n))),
            '最低': base_prices * (1 - np.abs(np.random.normal(0, 0.006, n))),
            '收盘': base_prices,
            '成交量': np.random.lognormal(16, 0.4, n)
        }, index=dates)

        return df


class FeatureEngineer:
    """特征工程类 - 使用Numba加速（修复版本）"""
    @staticmethod
    def create_features(df: pd.DataFrame, use_numba: bool = True) -> pd.DataFrame:
        """创建特征 - 完整修复版本"""
        df = df.copy()

        # 1. 基础数据验证和清理
        required_cols = ['开盘', '最高', '最低', '收盘', '成交量']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")

        # 清理异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')

        # 确保价格数据有效
        price_cols = ['开盘', '最高', '最低', '收盘']
        for col in price_cols:
            df[col] = df[col].replace(0, np.nan).fillna(method='ffill')
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        # 2. 基础收益率特征
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}'] = df['收盘'].pct_change(period)

        # 3. 移动平均特征（核心修复）
        for window in [5, 10, 20, 30, 60]:
            ma_col = f'ma_{window}'
            # 确保使用正确的滚动窗口计算
            df[ma_col] = df['收盘'].rolling(window=window, min_periods=1).mean()

            # 创建滞后特征（关键修复：使用shift避免未来数据）
            df[f'{ma_col}_lag1'] = df[ma_col].shift(1)
            df[f'price_vs_ma_{window}'] = (df['收盘'] / df[ma_col]) - 1

        # 4. 支撑阻力特征
        for window in [10, 20, 50]:
            support_col = f'support_{window}'
            resistance_col = f'resistance_{window}'

            df[support_col] = df['最低'].rolling(window=window, min_periods=1).min()
            df[resistance_col] = df['最高'].rolling(window=window, min_periods=1).max()

            # 滞后特征
            df[f'{support_col}_lag1'] = df[support_col].shift(1)
            df[f'{resistance_col}_lag1'] = df[resistance_col].shift(1)

            # 距离特征
            current_price = df['收盘'].shift(1)  # 使用滞后价格
            df[f'support_distance_{window}'] = (current_price - df[f'{support_col}_lag1']) / current_price
            df[f'resistance_distance_{window}'] = (df[f'{resistance_col}_lag1'] - current_price) / current_price

        # 5. 成交量特征
        df['volume_ma5'] = df['成交量'].rolling(5, min_periods=1).mean()
        df['volume_ma20'] = df['成交量'].rolling(20, min_periods=1).mean()

        df['volume_ratio_5'] = df['成交量'] / df['volume_ma5'].shift(1)
        df['volume_ratio_20'] = df['成交量'] / df['volume_ma20'].shift(1)
        df['volume_trend'] = (df['volume_ma5'] > df['volume_ma20']).astype(int)

        # 6. 波动率特征
        returns = df['收盘'].pct_change()
        for window in [5, 20]:
            df[f'volatility_{window}'] = returns.rolling(window=window, min_periods=1).std()

        # 7. K线形态特征（使用滞后数据）
        prev_open = df['开盘'].shift(1)
        prev_close = df['收盘'].shift(1)
        prev_high = df['最高'].shift(1)
        prev_low = df['最低'].shift(1)

        df['body_size'] = (prev_close - prev_open) / prev_open
        df['upper_shadow'] = (prev_high - np.maximum(prev_open, prev_close)) / prev_open
        df['lower_shadow'] = (np.minimum(prev_open, prev_close) - prev_low) / prev_open
        df['is_doji'] = (abs(df['body_size']) < 0.002).astype(int)

        # 8. 价格动量特征
        df['momentum_5'] = df['收盘'].shift(1) / df['收盘'].shift(6) - 1
        df['momentum_10'] = df['收盘'].shift(1) / df['收盘'].shift(11) - 1
        df['roc_5'] = (df['收盘'].shift(1) - df['收盘'].shift(6)) / df['收盘'].shift(6)

        # 9. RSI特征（简化稳定版本）
        try:
            # 使用更稳定的RSI计算
            delta = df['收盘'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_14_lag1'] = df['rsi_14'].shift(1)
        except Exception as e:
            logger.warning(f"RSI计算失败: {e}")
            df['rsi_14_lag1'] = 50  # 默认值

        # 10. 价格通道特征
        df['channel_high_20'] = df['最高'].rolling(20, min_periods=1).max().shift(1)
        df['channel_low_20'] = df['最低'].rolling(20, min_periods=1).min().shift(1)

        current_price_lag = df['收盘'].shift(1)
        channel_range = df['channel_high_20'] - df['channel_low_20']
        df['channel_position'] = (current_price_lag - df['channel_low_20']) / channel_range.replace(0, np.nan)

        # 11. 均线排列特征（修复版本）
        if all(col in df.columns for col in ['ma_5_lag1', 'ma_20_lag1', 'ma_60_lag1']):
            df['ma_alignment'] = (
                    (df['ma_5_lag1'] > df['ma_20_lag1']).astype(int) +
                    (df['ma_20_lag1'] > df['ma_60_lag1']).astype(int)
            )

        # 12. 目标变量定义（关键修复：使用滞后均线预测未来）
        # 使用滞后一期的均线作为基准，预测下一期是否突破
        if 'ma_10_lag1' in df.columns:
            df['target'] = (df['收盘'].shift(-1) > df['ma_10_lag1']).astype(int)
        else:
            # 备用目标：预测次日是否上涨
            df['target'] = (df['收盘'].shift(-1) > df['收盘']).astype(int)

        # 13. 清理最终数据
        # 移除包含未来信息的列（如果有）
        future_cols = [col for col in df.columns if 'shift_-' in str(col) or 'shift(-' in str(col)]
        if future_cols:
            df = df.drop(columns=future_cols)

        # 移除所有包含NaN的行
        initial_len = len(df)
        df = df.dropna()
        final_len = len(df)

        if initial_len - final_len > 0:
            logger.info(f"移除 {initial_len - final_len} 行包含NaN的数据")

        # 确保有足够的数据
        if len(df) < 60:
            raise ValueError(f"特征工程后数据不足，仅有 {len(df)} 行，需要至少60行")

        return df

    @staticmethod
    def select_features(df: pd.DataFrame, target: str = 'target', top_k: int = 15) -> Tuple[List[str], pd.DataFrame]:
        """选择特征"""
        feature_cols = [col for col in df.columns if col not in ['target', 'target_ma5', 'target_ma20']]

        # 优先选择安全的滞后特征
        safe_features = [col for col in feature_cols if
                         any(keyword in col for keyword in ['_lag', '_shift', 'return_'])]

        if len(safe_features) >= top_k:
            selected_features = safe_features[:top_k]
        else:
            selected_features = safe_features + [
                col for col in feature_cols if col not in safe_features
            ][:top_k - len(safe_features)]

        # 计算特征重要性
        X = df[selected_features].values
        y = df[target].values

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        best_features = importance_df.head(top_k)['feature'].tolist()

        return best_features, importance_df


class EnsembleLearner:
    """集成学习器 - 替换原有的单个模型训练"""

    def __init__(self, fixed_seed: int = 42):
        self.fixed_seed = fixed_seed
        self.base_models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()

    def create_base_models(self) -> Dict[str, Any]:
        """创建基础模型 - 修复模型参数"""
        base_models = {}

        try:
            # RandomForest - 修复参数
            base_models['RandomForest'] = RandomForestClassifier(
                n_estimators=200,  # 减少树数量提高速度
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=self.fixed_seed,
                n_jobs=-1
            )

            # XGBoost - 修复参数
            base_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.fixed_seed,
                eval_metric='logloss',
                n_jobs=-1
            )

            # LightGBM - 修复参数
            base_models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.fixed_seed,
                n_jobs=-1,
                verbose=-1  # 减少输出
            )

        except Exception as e:
            logger.error(f"创建基础模型失败: {e}")
            # 回退到简单模型
            base_models['RandomForest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=self.fixed_seed,
                n_jobs=-1
            )

        return base_models

    def create_voting_ensemble(self) -> Any:
        """创建投票集成模型"""
        try:
            base_models = self.create_base_models()
            if not base_models:
                logger.error("无法创建基础模型，回退到单个模型")
                return None

            # 使用前三个模型进行投票（RF, XGB, LGB）
            voting_models = [(name, model) for name, model in list(base_models.items())[:3]]

            ensemble = VotingClassifier(
                estimators=voting_models,
                **Config.ENSEMBLE_CONFIG['voting']
            )

            logger.info("成功创建投票集成模型")
            return ensemble

        except Exception as e:
            logger.error(f"创建投票集成模型失败: {e}")
            return None

    def create_stacking_ensemble(self) -> Any:
        """创建堆叠集成模型"""
        try:
            base_models = self.create_base_models()
            if not base_models:
                logger.error("无法创建基础模型，回退到单个模型")
                return None

            # 使用前三个模型进行堆叠
            stacking_models = [(name, model) for name, model in list(base_models.items())[:3]]

            ensemble = StackingClassifier(
                estimators=stacking_models,
                **Config.ENSEMBLE_CONFIG['stacking']
            )

            logger.info("成功创建堆叠集成模型")
            return ensemble

        except Exception as e:
            logger.error(f"创建堆叠集成模型失败: {e}")
            return None

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                       ensemble_type: str = 'voting') -> Dict[str, Any]:
        """训练集成模型"""
        results = {}

        try:
            # 选择集成类型
            if ensemble_type == 'voting':
                self.ensemble_model = self.create_voting_ensemble()
            elif ensemble_type == 'stacking':
                self.ensemble_model = self.create_stacking_ensemble()
            else:
                logger.warning(f"不支持的集成类型: {ensemble_type}，使用投票集成")
                self.ensemble_model = self.create_voting_ensemble()

            if self.ensemble_model is None:
                logger.error("集成模型创建失败，使用最佳单个模型")
                return self.train_individual_models(X_train, y_train)

            # 训练集成模型
            logger.info("开始训练集成模型...")
            start_time = time.time()

            self.ensemble_model.fit(X_train, y_train)

            training_time = time.time() - start_time
            logger.info(f"集成模型训练完成，耗时: {training_time:.2f}秒")

            # 同时训练基础模型用于比较
            individual_results = self.train_individual_models(X_train, y_train)

            results = {
                'ensemble_model': self.ensemble_model,
                'ensemble_type': ensemble_type,
                'training_time': training_time,
                'individual_models': individual_results
            }

        except Exception as e:
            logger.error(f"训练集成模型失败: {e}")
            # 回退到单个模型训练
            results = self.train_individual_models(X_train, y_train)

        return results

    def train_individual_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """训练单个模型（备用方案）"""
        models = {}

        try:
            base_models = self.create_base_models()

            for name, model in base_models.items():
                try:
                    model.fit(X_train, y_train)
                    models[name] = model
                    logger.info(f"单个模型 {name} 训练完成")
                except Exception as e:
                    logger.warning(f"单个模型 {name} 训练失败: {e}")

        except Exception as e:
            logger.error(f"所有单个模型训练失败: {e}")

        return models

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        elif self.base_models:
            # 使用最佳单个模型
            best_model = list(self.base_models.values())[0]
            return best_model.predict(X)
        else:
            raise ValueError("没有可用的模型进行预测")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X)
        elif self.base_models:
            best_model = list(self.base_models.values())[0]
            return best_model.predict_proba(X)
        else:
            raise ValueError("没有可用的模型进行概率预测")


class EnhancedModelTrainer:
    """增强的模型训练器 - 使用集成学习"""

    def __init__(self, fixed_seed: int = 42, use_ensemble: bool = True):
        self.fixed_seed = fixed_seed
        self.use_ensemble = use_ensemble
        self.ensemble_learner = EnsembleLearner(fixed_seed)
        self.scaler = StandardScaler()

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, ensemble_type: str = 'voting') -> Dict[str, Any]:
        """训练模型 - 优先使用集成学习"""
        if self.use_ensemble:
            logger.info("使用集成学习训练模型...")
            return self.ensemble_learner.train_ensemble(X_train, y_train, ensemble_type=ensemble_type)
        else:
            logger.info("使用单个模型训练...")
            return self.train_individual_models(X_train, y_train)

    def train_individual_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """训练单个模型（传统方法）"""
        models = {}

        base_models = self.ensemble_learner.create_base_models()
        for name, model in base_models.items():
            try:
                model.fit(X_train, y_train)
                models[name] = model
            except Exception as e:
                logger.warning(f"模型 {name} 训练失败: {e}")

        return {'individual_models': models}

    def evaluate_enhanced_models(self, ensemble_results: Dict, X_test: np.ndarray,
                                 y_test: np.ndarray, cv_results: Dict = None) -> Dict:
        """评估增强模型（集成+单个）"""
        all_results = {}

        try:
            # 评估集成模型
            if 'ensemble_model' in ensemble_results:
                ensemble_model = ensemble_results['ensemble_model']
                y_pred = ensemble_model.predict(X_test)
                y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                all_results['Ensemble'] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'model': ensemble_model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'ensemble_type': ensemble_results.get('ensemble_type', 'unknown')
                }

            # 评估单个模型
            individual_models = ensemble_results.get('individual_models', {})
            for name, model in individual_models.items():
                try:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_pred_proba)

                    all_results[name] = {
                        'accuracy': accuracy,
                        'auc': auc_score,
                        'model': model,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                except Exception as e:
                    logger.warning(f"评估单个模型 {name} 失败: {e}")

        except Exception as e:
            logger.error(f"评估增强模型失败: {e}")

        return all_results

    def cross_validate(self, models: Dict, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> Dict:
        """交叉验证"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}

        for name, model in models.items():
            fold_scores = []
            fold_auc_scores = []

            for train_idx, test_idx in tscv.split(X):
                if max(train_idx) >= min(test_idx):
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)

                    y_pred_proba = model_clone.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, model_clone.predict(X_test))
                    auc_score = roc_auc_score(y_test, y_pred_proba)

                    fold_scores.append(accuracy)
                    fold_auc_scores.append(auc_score)

                except Exception:
                    continue

            if fold_scores:
                cv_results[name] = {
                    'accuracy_mean': np.mean(fold_scores),
                    'accuracy_std': np.std(fold_scores),
                    'auc_mean': np.mean(fold_auc_scores),
                    'auc_std': np.std(fold_auc_scores),
                    'n_successful_folds': len(fold_scores)
                }

        return cv_results


class OptimizedBacktestEngine:
    """优化回测引擎 - 修复版本"""

    def __init__(self, transaction_cost: float = 0.001, slippage: float = 0.002):
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def identify_trades_optimized(self, df_strategy: pd.DataFrame,
                                  min_hold_days: int = 5,
                                  prob_threshold: float = 0.55) -> List[Dict]:
        """优化交易识别 - 减少过度交易"""
        trades = []
        in_position = False
        entry_price = 0
        entry_date = None
        last_trade_date = None

        for i in range(1, len(df_strategy)):
            current_date = df_strategy.index[i]
            current_price = df_strategy['收盘'].iloc[i]
            current_prob = df_strategy['probability'].iloc[i]
            current_signal = df_strategy['signal'].iloc[i]

            # 检查最小持仓时间
            days_since_last_trade = 0
            if last_trade_date:
                days_since_last_trade = (current_date - last_trade_date).days

            # 开仓条件：强烈买入信号 + 满足最小间隔
            if (not in_position and
                    current_signal == 1 and
                    current_prob >= prob_threshold and
                    (last_trade_date is None or days_since_last_trade >= min_hold_days)):

                in_position = True
                entry_price = current_price * (1 + self.slippage)
                entry_date = current_date

            # 平仓条件：强烈卖出信号 或 达到止损
            elif in_position:
                should_exit = False
                exit_reason = ""

                # 强烈卖出信号
                if current_signal == 0 and current_prob <= (1 - prob_threshold):
                    should_exit = True
                    exit_reason = "卖出信号"
                # 止损条件（示例：下跌8%）
                elif (current_price - entry_price) / entry_price < -0.08:
                    should_exit = True
                    exit_reason = "止损"
                # 最小持仓时间检查
                elif (current_date - entry_date).days >= min_hold_days:
                    # 盈利保护：如果盈利超过5%，在概率转弱时退出
                    profit = (current_price - entry_price) / entry_price
                    if profit > 0.05 and current_prob < 0.5:
                        should_exit = True
                        exit_reason = "盈利保护"

                if should_exit:
                    in_position = False
                    exit_price = current_price * (1 - self.slippage)
                    trade_return = (exit_price - entry_price) / entry_price - 2 * self.transaction_cost

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'days_held': (current_date - entry_date).days,
                        'type': '盈利' if trade_return > 0 else '亏损',
                        'exit_reason': exit_reason,
                        'entry_prob': df_strategy.loc[entry_date, 'probability'],
                        'exit_prob': current_prob
                    })

                    last_trade_date = current_date

        return trades

    def calculate_performance_metrics_fixed(self, df_strategy: pd.DataFrame, trades: List[Dict]) -> Dict:
        """修复性能指标计算"""
        if not trades:
            return {}

        # 计算策略累计收益 - 修复版本
        initial_capital = 1.0
        capital = initial_capital

        for trade in trades:
            capital = capital * (1 + trade['return'])

        total_return = (capital - initial_capital) / initial_capital

        # 市场基准收益
        market_return = (df_strategy['收盘'].iloc[-1] / df_strategy['收盘'].iloc[0]) - 1

        # 其他指标计算
        winning_trades = [t for t in trades if t['return'] > 0]
        losing_trades = [t for t in trades if t['return'] <= 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # 计算收益指标
        returns = [t['return'] for t in trades]
        avg_return = np.mean(returns) if returns else 0
        avg_win_return = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
        avg_loss_return = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0

        # 盈利因子
        gross_profit = sum([t['return'] for t in winning_trades if t['return'] > 0])
        gross_loss = abs(sum([t['return'] for t in losing_trades if t['return'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 计算平均持仓天数 - 修复版本
        if trades and 'days_held' in trades[0]:
            avg_holding_days = np.mean([t.get('days_held', 0) for t in trades])
        else:
            # 如果没有days_held字段，从日期计算
            holding_days = []
            for trade in trades:
                if 'entry_date' in trade and 'exit_date' in trade:
                    entry_date = trade['entry_date']
                    exit_date = trade['exit_date']
                    if hasattr(entry_date, 'date') and hasattr(exit_date, 'date'):
                        days_held = (exit_date - entry_date).days
                        holding_days.append(days_held)
            avg_holding_days = np.mean(holding_days) if holding_days else 0

        # 波动率和夏普比率 - 基于策略收益率计算
        strategy_returns = df_strategy['strategy_return'].dropna()
        if len(strategy_returns) > 0:
            strategy_volatility = strategy_returns.std() * np.sqrt(252)
            # 年化收益率基于实际交易收益计算
            annualized_return = (1 + total_return) ** (252 / len(df_strategy)) - 1
            sharpe_ratio = annualized_return / strategy_volatility if strategy_volatility > 0 else 0
        else:
            strategy_volatility = 0
            annualized_return = 0
            sharpe_ratio = 0

        # 最大回撤 - 基于策略累计收益
        max_drawdown = df_strategy['strategy_drawdown'].min() if 'strategy_drawdown' in df_strategy.columns else 0

        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'avg_return': avg_return,
            'avg_win_return': avg_win_return,
            'avg_loss_return': avg_loss_return,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'annualized_return': annualized_return,
            'volatility': strategy_volatility,
            'avg_holding_days': avg_holding_days  # 修复：添加平均持仓天数
        }

    def generate_signal_info_improved(self, df_strategy: pd.DataFrame) -> Dict:
        """改进的信号生成逻辑 - 使用概率阈值，信号日期和价格为最后一次交易"""
        if len(df_strategy) == 0:
            return {
                'current_signal': "数据不足",
                'explanation': "数据量不足无法生成信号",
                'date': None,
                'price': 0,
                'current_price': 0
            }

        current_prob = df_strategy['probability'].iloc[-1]
        prev_signal = df_strategy['signal'].iloc[-2] if len(df_strategy) > 1 else 0

        # 使用概率阈值来确定信号
        buy_threshold = 0.6  # 买入阈值：概率>60%
        sell_threshold = 0.4  # 卖出阈值：概率<40%

        if current_prob >= buy_threshold:
            signal_type = "买入"
            explanation = f"强烈买入信号 (概率: {current_prob:.2%})"
        elif current_prob <= sell_threshold:
            signal_type = "卖出"
            explanation = f"强烈卖出信号 (概率: {current_prob:.2%})"
        elif prev_signal == 1:
            signal_type = "持仓"
            explanation = f"继续持仓 (当前概率: {current_prob:.2%})"
        else:
            signal_type = "观望"
            explanation = f"观望状态 (概率: {current_prob:.2%})"

        # 获取最后一次交易的日期和价格
        last_trade_date = None
        last_trade_price = 0

        # 查找最后一次买卖信号
        buy_signals = df_strategy[df_strategy['buy_signal'] == True]
        sell_signals = df_strategy[df_strategy['sell_signal'] == True]

        if not buy_signals.empty and not sell_signals.empty:
            last_buy = buy_signals.index[-1] if not buy_signals.empty else None
            last_sell = sell_signals.index[-1] if not sell_signals.empty else None

            if last_buy and last_sell:
                # 取最近的信号
                if last_buy > last_sell:
                    last_trade_date = last_buy
                    last_trade_price = df_strategy.loc[last_buy, '收盘']
                else:
                    last_trade_date = last_sell
                    last_trade_price = df_strategy.loc[last_sell, '收盘']
            elif last_buy:
                last_trade_date = last_buy
                last_trade_price = df_strategy.loc[last_buy, '收盘']
            elif last_sell:
                last_trade_date = last_sell
                last_trade_price = df_strategy.loc[last_sell, '收盘']

        # 如果没有交易信号，使用最新数据
        if last_trade_date is None:
            last_trade_date = df_strategy.index[-1]
            last_trade_price = df_strategy['收盘'].iloc[-1]

        return {
            'current_signal': signal_type,
            'explanation': explanation,
            'date': last_trade_date,
            'price': last_trade_price,
            'current_price': df_strategy['收盘'].iloc[-1],
            'probability': current_prob
        }

    def run_backtest(self, df: pd.DataFrame, model_results: Dict, features: List[str],
                     scaler: StandardScaler, target: str = 'target') -> Optional[Dict]:
        """运行回测 - 修复策略收益计算"""
        if not model_results:
            return None

        # 选择最佳模型
        best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = model_results[best_model_name]['model']

        # 准备数据
        X = df[features].values
        X_scaled = scaler.transform(X)

        # 预测
        predictions = best_model.predict(X_scaled)
        probabilities = best_model.predict_proba(X_scaled)[:, 1]

        # 创建策略数据框
        df_strategy = df.copy()
        df_strategy = df_strategy.iloc[len(df_strategy) - len(predictions):]
        df_strategy['signal'] = predictions
        df_strategy['probability'] = probabilities

        # 计算收益率 - 修复版本
        df_strategy['daily_return'] = df_strategy['收盘'].pct_change().fillna(0)
        df_strategy['next_daily_return'] = df_strategy['daily_return'].shift(-1).fillna(0)

        # 策略逻辑
        df_strategy['position'] = df_strategy['signal']
        df_strategy['position_change'] = df_strategy['position'].diff()
        df_strategy['buy_signal'] = (df_strategy['position_change'] == 1)
        df_strategy['sell_signal'] = (df_strategy['position_change'] == -1)

        # 策略收益率计算 - 修复版本
        # 使用更准确的方法计算策略收益率
        df_strategy['strategy_return'] = 0.0

        # 初始化变量
        cash = 1.0  # 初始资金为1
        position_value = 0.0
        in_position = False

        for i in range(1, len(df_strategy)):
            current_signal = df_strategy['signal'].iloc[i]
            prev_signal = df_strategy['signal'].iloc[i - 1]
            daily_return = df_strategy['daily_return'].iloc[i]

            # 开仓信号
            if not in_position and current_signal == 1:
                in_position = True
                position_value = cash
                cash = 0.0
                # 扣除交易成本
                df_strategy.loc[df_strategy.index[i], 'strategy_return'] = -self.transaction_cost
            # 平仓信号
            elif in_position and current_signal == 0:
                in_position = False
                cash = position_value * (1 + daily_return) * (1 - self.transaction_cost)
                position_value = 0.0
                df_strategy.loc[df_strategy.index[i], 'strategy_return'] = daily_return - self.transaction_cost
            # 持仓中
            elif in_position:
                position_value = position_value * (1 + daily_return)
                df_strategy.loc[df_strategy.index[i], 'strategy_return'] = daily_return
            # 空仓中
            else:
                df_strategy.loc[df_strategy.index[i], 'strategy_return'] = 0.0

        # 计算累计收益 - 修复版本
        df_strategy['cumulative_market'] = (1 + df_strategy['next_daily_return']).cumprod()
        df_strategy['cumulative_strategy'] = (1 + df_strategy['strategy_return']).cumprod()

        # 计算回撤
        df_strategy['strategy_cummax'] = df_strategy['cumulative_strategy'].cummax()
        df_strategy['strategy_drawdown'] = (
                (df_strategy['cumulative_strategy'] - df_strategy['strategy_cummax']) /
                df_strategy['strategy_cummax']
        )

        # 识别交易和计算指标 - 使用优化版本
        trades = self.identify_trades_optimized(df_strategy)
        metrics = self.calculate_performance_metrics_fixed(df_strategy, trades)

        # 生成交易信号 - 使用改进版本
        signal_info = self.generate_signal_info_improved(df_strategy)

        return {
            'df_strategy': df_strategy,
            'trades': trades,
            'metrics': metrics,
            'signal_info': signal_info,
            'best_model': best_model_name,
            'buy_dates': df_strategy.index[df_strategy['buy_signal']],
            'sell_dates': df_strategy.index[df_strategy['sell_signal']]
        }


class EnhancedMarketReplayPredictor:
    """增强的市场回放预测器 - 使用集成学习和Numba加速（修复版本 + 前复权数据）"""

    def __init__(self, symbol="000001", start_date=None, end_date=None,
                 test_start_date="20240101", fixed_seed=42,
                 transaction_cost=0.001, slippage=0.002, use_ensemble=True,
                 use_numba=True):

        self.symbol = symbol
        # 如果没有指定开始日期，默认使用3年前的日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y%m%d")
        self.start_date = start_date

        # 结束日期参数，默认当天
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        self.end_date = end_date

        # 回放模式特有参数
        self.test_start_date = test_start_date
        self.fixed_seed = fixed_seed
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.use_ensemble = use_ensemble
        self.use_numba = use_numba and NUMBA_AVAILABLE  # Numba加速开关

        # 初始化组件
        self.model_trainer = EnhancedModelTrainer(fixed_seed, use_ensemble)
        self.backtest_engine = OptimizedBacktestEngine(transaction_cost, slippage)

        self.best_model = None
        self.data_quality_report = {}
        self.stock_name = StockDataFetcher.get_stock_name(symbol)

        # 设置随机种子确保结果一致性
        np.random.seed(fixed_seed)
        random.seed(fixed_seed)

    def get_stock_data(self, use_cache=True):
        """获取股票数据 - 使用前复权数据"""
        # 确定分析模式
        analysis_mode = "replay" if self.end_date != datetime.now().strftime("%Y%m%d") else "current"

        if use_cache:
            cached_data = CacheManager.load_from_cache(self.symbol, self.start_date, self.end_date, analysis_mode)
            if cached_data is not None:
                st.success(f"从缓存加载 {self.stock_name}({self.symbol}) 数据 (回放模式: {self.end_date})")
                return cached_data

        # 获取真实数据 - 使用前复权数据
        df = StockDataFetcher.fetch_stock_data_enhanced(self.symbol, self.start_date, self.end_date)

        if df is None or len(df) < 100:
            st.warning(f"股票 {self.symbol} 数据获取失败或数据量不足，使用模拟数据")
            df = StockDataFetcher.generate_simulated_data(self.start_date, self.end_date, self.fixed_seed)

        # 数据质量检查
        self._check_data_quality(df)

        if use_cache:
            CacheManager.save_to_cache(self.symbol, self.start_date, df, self.end_date, analysis_mode)

        mode_text = "回放模式" if analysis_mode == "replay" else "实时模式"
        st.info(
            f"✅成功获取 {self.stock_name}({self.symbol}) 前复权数据: {len(df)} 条记录 ({mode_text}: {self.start_date}到{self.end_date})")
        return df

    def _check_data_quality(self, df):
        """检查数据质量"""
        # 确保所有必需的列都存在
        required_columns = ['开盘', '最高', '最低', '收盘', '成交量']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"数据缺少必需列: {missing_columns}")
            # 尝试修复：如果收盘价列存在但其他价格列缺失，使用收盘价填充
            if '收盘' in df.columns:
                for col in missing_columns:
                    if col in ['开盘', '最高', '最低']:
                        df[col] = df['收盘']
                    elif col == '成交量':
                        df[col] = 1000000  # 默认成交量

        # 检查数据完整性
        missing_values = df.isnull().sum()
        price_anomalies = (df['收盘'] <= 0).sum() if '收盘' in df.columns else 0
        volume_anomalies = (df['成交量'] <= 0).sum() if '成交量' in df.columns else 0
        date_diff = df.index.to_series().diff().dt.days if not df.empty else pd.Series()
        gap_days = date_diff[date_diff > 1] if not date_diff.empty else []

        self.data_quality_report = {
            'missing_columns': missing_columns,
            'missing_values': missing_values.to_dict() if not missing_values.empty else {},
            'price_anomalies': price_anomalies,
            'volume_anomalies': volume_anomalies,
            'date_gaps': len(gap_days)
        }

    def prepare_data(self, df, features, target='target'):
        """准备数据 - 使用固定日期分割（回放模式特有）"""
        # 确保特征列存在
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]

        if missing_features:
            logger.warning(f"以下特征列不存在，将被忽略: {missing_features}")

        if not available_features:
            st.error("没有可用的特征列进行训练")
            return None, None, None, None

        X = df[available_features].values
        y = df[target].values if target in df.columns else np.zeros(len(df))

        # 数据标准化
        X_scaled = self.model_trainer.scaler.fit_transform(X)

        # 使用固定日期分割（回放模式）
        test_start_dt = pd.to_datetime(self.test_start_date)

        # 找到最接近的日期索引
        try:
            test_start_idx = df.index.get_indexer([test_start_dt], method='nearest')[0]
        except:
            test_start_idx = -1

        if test_start_idx <= 0 or test_start_idx >= len(X_scaled):
            # 如果固定日期无效，使用比例分割
            test_size = 0.2
            split_idx = int(len(X_scaled) * (1 - test_size))
            st.info(f"使用比例分割: 训练集{split_idx}条, 测试集{len(X_scaled) - split_idx}条")
        else:
            split_idx = test_start_idx
            st.info(f"使用固定日期分割: 训练集{split_idx}条, 测试集{len(X_scaled) - split_idx}条")
        # 确保训练集和测试集的时间顺序
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def run_enhanced_replay_analysis(self, use_cache=True, ensemble_type='voting'):
        """运行增强的回放分析"""
        try:
            # 获取数据
            df = self.get_stock_data(use_cache)

            if df is None or len(df) == 0:
                st.error("无法获取股票数据")
                return None

            # 创建特征 - 使用Numba加速
            start_time = time.time()
            try:
                df = FeatureEngineer.create_features(df, use_numba=self.use_numba)
                feature_time = time.time() - start_time

            except Exception as e:
                st.error(f"特征工程失败: {e}")
                # 尝试不使用Numba加速
                st.info("尝试使用非加速版本的特征工程...")
                df = FeatureEngineer.create_features(df, use_numba=False)
                feature_time = time.time() - start_time
                st.info(f"特征工程完成 (非加速) - 耗时: {feature_time:.2f}秒")

            if len(df) == 0:
                st.error("特征工程后数据为空")
                return None

            # 选择特征
            best_features, feature_importance = FeatureEngineer.select_features(df, top_k=15)

            # 准备数据（使用回放模式特有的固定日期分割）
            X_train, X_test, y_train, y_test = self.prepare_data(df, best_features)

            if X_train is None:
                return None

            if len(X_test) == 0 or len(y_test) == 0:
                # 如果测试集为空，调整分割比例
                from sklearn.model_selection import train_test_split
                X_all = np.vstack([X_train, X_test])
                y_all = np.hstack([y_train, y_test])
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.15, random_state=self.fixed_seed, stratify=y_all
                )

            if len(X_test) == 0 or len(y_test) == 0:
                st.error("调整后测试集仍为空，无法进行分析")
                return None

            # 使用集成学习训练模型
            ensemble_results = self.model_trainer.train_models(X_train, y_train, ensemble_type=ensemble_type)

            if not ensemble_results:
                st.error("模型训练失败，无法继续分析")
                return None

            # 增强的交叉验证
            X_all = np.vstack([X_train, X_test])
            y_all = np.hstack([y_train, y_test])
            cv_results = self.model_trainer.cross_validate(
                {**ensemble_results.get('individual_models', {}),
                 **({'Ensemble': ensemble_results['ensemble_model']} if 'ensemble_model' in ensemble_results else {})},
                X_all, y_all
            )

            # 评估模型
            results = self.model_trainer.evaluate_enhanced_models(
                ensemble_results, X_test, y_test, cv_results
            )

            if not results:
                st.error("所有模型评估失败")
                return None

            # 选择最佳模型
            best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            self.best_model = results[best_model_name]['model']

            # 回测策略
            strategy_results = self.backtest_engine.run_backtest(
                df, results, best_features, self.model_trainer.scaler
            )

            return {
                'symbol': self.symbol,
                'stock_name': self.stock_name,
                'best_model': best_model_name,
                'best_accuracy': results[best_model_name]['accuracy'],
                'best_auc': results[best_model_name]['auc'],
                'all_results': results,
                'feature_importance': feature_importance,
                'strategy_results': strategy_results,
                'data_points': len(df),
                'y_test': y_test,
                'data_quality': self.data_quality_report,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'analysis_mode': 'replay',
                'fixed_seed': self.fixed_seed,
                'using_ensemble': self.use_ensemble,
                'using_numba': self.use_numba,
                'feature_time': feature_time,
                'ensemble_type': ensemble_results.get('ensemble_type', 'individual'),
                'cv_results': cv_results
            }

        except Exception as e:
            st.error(f"增强分析错误: {e}")
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    def run_simplified_analysis(self, use_cache=True):
        """运行简化分析 - 确保基础功能稳定"""
        try:
            # 1. 获取数据
            df = self.get_stock_data(use_cache)
            if df is None or len(df) < 60:
                st.error("数据获取失败或数据量不足")
                return None

            # 2. 创建特征（使用修复版本）
            df = FeatureEngineer.create_features(df, use_numba=False)  # 暂时关闭numba

            if len(df) == 0:
                st.error("特征工程后数据为空")
                return None

            # 3. 选择特征
            feature_cols = [col for col in df.columns if col.startswith(('ma_', 'return_')) and col != 'target']
            if not feature_cols:
                st.error("无有效特征可用")
                return None

            # 4. 准备数据
            X = df[feature_cols].values
            y = df['target'].values

            # 简单数据分割
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # 5. 训练简化模型
            trainer = SimplifiedModelTrainer(self.fixed_seed)
            model = trainer.train_simple_model(X_train, y_train)

            if model is None:
                return None

            # 6. 评估模型
            results = trainer.evaluate_model(model, X_test, y_test)

            return {
                'symbol': self.symbol,
                'stock_name': self.stock_name,
                'model': model,
                'results': results,
                'feature_cols': feature_cols,
                'data_points': len(df)
            }

        except Exception as e:
            st.error(f"简化分析失败: {e}")
            return None

    def run_analysis(self, use_cache=True, use_simple=False):
        """运行分析 - 可选择简化版本"""
        if use_simple:
            return self.run_simplified_analysis(use_cache)
        else:
            return self.run_enhanced_replay_analysis(use_cache)


class SimplifiedModelTrainer:
    """简化模型训练器 - 确保基础功能稳定"""

    def __init__(self, fixed_seed: int = 42):
        self.fixed_seed = fixed_seed
        self.scaler = StandardScaler()

    def train_simple_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """训练简化模型 - 确保基础功能"""
        try:
            # 使用最稳定的随机森林
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.fixed_seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            st.error(f"模型训练失败: {e}")
            return None

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型"""
        if model is None:
            return {}

        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        except Exception as e:
            st.error(f"模型评估失败: {e}")
            return {}


# 可视化函数保持不变
def plot_results(results, y_test, symbol, stock_name, cv_results=None):
    """绘制结果图表"""
    setup_chinese_font()

    if not results:
        st.error("没有可用的模型结果进行绘图")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('black')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
    models = list(results.keys())

    # 1. 模型比较（测试准确率 + CV准确率）
    test_accuracies = [results[model]['accuracy'] for model in models]

    # 如果有CV结果，显示误差条
    if cv_results:
        cv_means = [cv_results.get(model, {}).get('accuracy_mean', 0) for model in models]
        cv_stds = [cv_results.get(model, {}).get('accuracy_std', 0) for model in models]

        x_pos = np.arange(len(models))
        width = 0.35

        bars1 = axes[0, 0].bar(x_pos - width / 2, test_accuracies, width,
                               label='测试准确率', color=colors[0], alpha=0.8)
        bars2 = axes[0, 0].bar(x_pos + width / 2, cv_means, width,
                               yerr=cv_stds, label='CV准确率', color=colors[1], alpha=0.8, capsize=5)

        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=45)
    else:
        bars = axes[0, 0].bar(models, test_accuracies, color=colors[:len(models)])

    axes[0, 0].set_title(f'{stock_name}({symbol}) - 模型准确率比较', color='white', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('准确率', color='white')
    axes[0, 0].tick_params(colors='white')
    axes[0, 0].legend(facecolor='black', edgecolor='white', labelcolor='white')
    axes[0, 0].grid(True, alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(test_accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', color='white', fontweight='bold')

    # 2. 特征重要性（最佳模型）
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    if hasattr(results[best_model_name]['model'], 'feature_importances_'):
        importances = results[best_model_name]['model'].feature_importances_
        if len(importances) > 0:
            indices = np.argsort(importances)[::-1][:min(10, len(importances))]

            axes[0, 1].bar(range(len(indices)), importances[indices], color='#FFA07A')
            axes[0, 1].set_title(f'{best_model_name} - 特征重要性 (Top 10)', color='white', fontsize=14,
                                 fontweight='bold')
            axes[0, 1].set_xticks(range(len(indices)))
            axes[0, 1].tick_params(colors='white')
            axes[0, 1].grid(True, alpha=0.3)

    # 3. ROC曲线
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.3f})',
                        color=colors[i % len(colors)], linewidth=2)

    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('假正率', color='white')
    axes[1, 0].set_ylabel('真正率', color='white')
    axes[1, 0].set_title(f'{stock_name}({symbol}) - ROC曲线', color='white', fontsize=14, fontweight='bold')
    axes[1, 0].legend(facecolor='black', edgecolor='white', labelcolor='white')
    axes[1, 0].tick_params(colors='white')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 预测概率分布
    best_result = results[best_model_name]
    if len(y_test) > 0:
        axes[1, 1].hist(best_result['probabilities'][y_test == 0], bins=20, alpha=0.7,
                        label='实际下跌', color='#FF6B6B', edgecolor='white')
        axes[1, 1].hist(best_result['probabilities'][y_test == 1], bins=20, alpha=0.7,
                        label='实际上涨', color='#4ECDC4', edgecolor='white')
        axes[1, 1].axvline(x=0.5, color='white', linestyle='--', alpha=0.8, label='决策阈值')
        axes[1, 1].set_xlabel('预测概率', color='white')
        axes[1, 1].set_ylabel('频次', color='white')
        axes[1, 1].set_title(f'{stock_name}({symbol}) - 预测概率分布', color='white', fontsize=14, fontweight='bold')
        axes[1, 1].legend(facecolor='black', edgecolor='white', labelcolor='white')
        axes[1, 1].tick_params(colors='white')
        axes[1, 1].grid(True, alpha=0.3)

    # 设置所有子图的背景色和边框色
    for ax in axes.flat:
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.tight_layout()
    return fig


def plot_strategy_performance(df_strategy, symbol, stock_name, buy_dates, sell_dates, market_return, total_return):
    """绘制策略表现图"""
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.patch.set_facecolor('black')

    ax1.set_facecolor('black')
    ax1.plot(df_strategy.index, df_strategy['cumulative_market'], label=f'市场收益 ({market_return:.2%})', linewidth=2,
             alpha=0.7, color='#FF6B6B')
    ax1.plot(df_strategy.index, df_strategy['cumulative_strategy'], label=f'策略收益 ({total_return:.2%})', linewidth=2,
             color='#4ECDC4')

    if len(buy_dates) > 0:
        buy_values = df_strategy.loc[buy_dates, 'cumulative_strategy']
        ax1.scatter(buy_dates, buy_values, color='#FF0000', marker='^', s=100,
                    label='买入点', zorder=5, alpha=0.8)

    if len(sell_dates) > 0:
        sell_values = df_strategy.loc[sell_dates, 'cumulative_strategy']
        ax1.scatter(sell_dates, sell_values, color='#00FF00', marker='v', s=50,
                    label='卖出点', zorder=5, alpha=0.8)

    ax1.set_title(f'{stock_name}({symbol}) - 累计收益对比 (带买卖点)', fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel('累计收益', color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax1.grid(True, alpha=0.3)

    for spine in ax1.spines.values():
        spine.set_color('white')

    ax2.set_facecolor('black')
    ax2.fill_between(df_strategy.index, df_strategy['strategy_drawdown'] * 100, 0,
                     alpha=0.3, color='#FF6B6B', label='回撤')

    if len(buy_dates) > 0:
        buy_drawdown = df_strategy.loc[buy_dates, 'strategy_drawdown'] * 100
        ax2.scatter(buy_dates, buy_drawdown, color='#FF0000', marker='^', s=80,
                    zorder=5, alpha=0.8, label='买入点')

    if len(sell_dates) > 0:
        sell_drawdown = df_strategy.loc[sell_dates, 'strategy_drawdown'] * 100
        ax2.scatter(sell_dates, sell_drawdown, color='#00FF00', marker='v', s=50,
                    zorder=3, alpha=0.8, label='卖出点')

    ax2.set_title(f'{stock_name}({symbol}) - 策略回撤', fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel('回撤 (%)', color='white')
    ax2.set_xlabel('日期', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax2.grid(True, alpha=0.3)

    for spine in ax2.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    return fig


def plot_price_with_signals(df_strategy, symbol, stock_name, buy_dates, sell_dates):
    """绘制价格走势与买卖信号"""
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    fig.patch.set_facecolor('black')

    ax1.set_facecolor('black')
    ax1.plot(df_strategy.index, df_strategy['收盘'], label='收盘价', linewidth=2, color='#45B7D1')

    if len(buy_dates) > 0:
        buy_prices = df_strategy.loc[buy_dates, '收盘']
        ax1.scatter(buy_dates, buy_prices, color='#FF0000', marker='^', s=100,
                    label='买入点', zorder=5, alpha=0.8)

    if len(sell_dates) > 0:
        sell_prices = df_strategy.loc[sell_dates, '收盘']
        ax1.scatter(sell_dates, sell_prices, color='#00FF00', marker='v', s=50,
                    label='卖出点', zorder=3, alpha=0.8)

    ax1.set_title(f'{stock_name}({symbol}) - 价格走势与买卖点', fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel('价格', color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax1.grid(True, alpha=0.3)

    for spine in ax1.spines.values():
        spine.set_color('white')

    ax2.set_facecolor('black')
    ax2.plot(df_strategy.index, df_strategy['probability'], label='买入信号概率', color='#FFA07A', linewidth=2)
    ax2.axhline(y=0.5, color='white', linestyle='--', alpha=0.7, label='决策阈值')

    if len(buy_dates) > 0:
        buy_probs = df_strategy.loc[buy_dates, 'probability']
        ax2.scatter(buy_dates, buy_probs, color='#FF0000', marker='^', s=80, zorder=5, label='买入点')

    if len(sell_dates) > 0:
        sell_probs = df_strategy.loc[sell_dates, 'probability']
        ax2.scatter(sell_dates, sell_probs, color='#00FF00', marker='v', s=50, zorder=3, label='卖出点')

    ax2.set_title(f'{stock_name}({symbol}) - 模型信号概率', fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel('概率', color='white')
    ax2.set_xlabel('日期', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax2.grid(True, alpha=0.3)

    for spine in ax2.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    return fig


def display_enhanced_results(results):
    """增强的结果展示 - 修改信号说明为最近一次买卖操作"""
    strategy = results['strategy_results']
    metrics = strategy['metrics']

    # 创建更详细的指标卡片
    col1, col2, col3, col4 = st.columns(4)

    # 交易质量分析
    trades = strategy.get('trades', [])
    if trades:
        winning_trades = [t for t in trades if t['return'] > 0]
        losing_trades = [t for t in trades if t['return'] <= 0]
        max_return = max([t.get('return', 0) * 100 for t in trades]) if trades else 0
        min_return = min([t.get('return', 0) * 100 for t in trades]) if trades else 0
        avg_return = np.mean([t.get('return', 0) * 100 for t in trades]) if trades else 0

        # 统一计算盈亏比
        if losing_trades:
            # 方法1: 总盈利/总亏损 (盈利因子)
            gross_profit = sum([t['return'] for t in winning_trades])
            gross_loss = abs(sum([t['return'] for t in losing_trades]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # 方法2: 平均盈利/平均亏损
            avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t['return'] for t in losing_trades])) if losing_trades else 0
            avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        else:
            profit_factor = float('inf')
            avg_win_loss_ratio = float('inf')

        # 获取最近一次买卖操作
        date_str = "/"
        last_operation = "/"

        # 从策略数据中获取买卖信号
        df_strategy = strategy.get('df_strategy', pd.DataFrame())
        if not df_strategy.empty:
            # 查找最近的买入或卖出信号
            buy_signals = df_strategy[df_strategy['buy_signal'] == True]
            sell_signals = df_strategy[df_strategy['sell_signal'] == True]

            if not buy_signals.empty and not sell_signals.empty:
                last_buy = buy_signals.index[-1] if not buy_signals.empty else None
                last_sell = sell_signals.index[-1] if not sell_signals.empty else None

                # 取最近的信号
                if last_buy and last_sell:
                    if last_buy > last_sell:
                        last_operation = "买入"
                        last_date = last_buy
                    else:
                        last_operation = "卖出"
                        last_date = last_sell
                elif last_buy:
                    last_operation = "买入"
                    last_date = last_buy
                elif last_sell:
                    last_operation = "卖出"
                    last_date = last_sell
                else:
                    last_operation = "/"
                    last_date = None

                if last_date:
                    # 格式化日期
                    if hasattr(last_date, 'strftime'):
                        date_str = last_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(last_date)

            elif not buy_signals.empty:
                last_buy = buy_signals.index[-1]
                date_str = last_buy.strftime('%Y-%m-%d') if hasattr(last_buy, 'strftime') else str(last_buy)
            elif not sell_signals.empty:
                last_sell = sell_signals.index[-1]
                date_str = last_sell.strftime('%Y-%m-%d') if hasattr(last_sell, 'strftime') else str(last_sell)

        with col1:
            st.metric("累计收益率", f"{metrics['total_return']:.2%}")
            st.metric("年化收益率", f"{metrics.get('annualized_return', 0):.2%}")
            winnings = metrics.get('winning_trades', 0)
            losings = metrics.get('total_trades', 0) - winnings
            st.metric("总交易次数", f"{metrics['total_trades']}({winnings}/{losings})")

        with col2:
            st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
            st.metric("最大单笔收益", f"{max_return:.2f}%")

        with col3:
            st.metric("胜率", f"{metrics['win_rate']:.2%}")
            st.metric("总盈亏比", f"{profit_factor:.2f}")
            st.metric("最大单笔亏损", f"{min_return:.2f}%")

        with col4:
            st.metric(date_str, last_operation)
            st.metric("平均收益率", f"{avg_return:.2f}%")
            st.metric("平均持仓天数", f"{metrics.get('avg_holding_days', 0):.1f}天")

        # 计算显示用的平均值
        avg_win_display = np.mean([t['return'] for t in winning_trades]) * 100 if winning_trades else 0
        avg_loss_display = np.mean([t['return'] for t in losing_trades]) * 100 if losing_trades else 0

        # 显示两种盈亏比计算方法，让用户了解差异
        st.info(
            f"📊 交易质量: "
            f"平均盈利 {avg_win_display:.2f}% | "
            f"平均亏损 {avg_loss_display:.2f}% | "
            f"平均盈亏比 {avg_win_loss_ratio:.2f} 　"
            f"💡 信号说明: **{strategy['signal_info']['explanation']}**"
        )

# 主函数保持不变
def main():
    """主函数 - Streamlit应用"""
    st.set_page_config(
        page_title="股价预测分析系统 - 集成学习优化版 + Numba加速 + 前复权数据",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义CSS样式
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #0e1117;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .ensemble-badge {
        background-color: #4ECDC4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .numba-badge {
        background-color: #FF6B6B;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .adjusted-badge {
        background-color: #45B7D1;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # 使用session_state跟踪之前的选择
    if 'prev_analysis_mode' not in st.session_state:
        st.session_state.prev_analysis_mode = None

    # 分析模式选择
    analysis_mode = st.sidebar.radio("选择分析模式", ["单支分析", "批量分析"])
    replay_mode = st.sidebar.checkbox("启用回放模式", value=False,
                                      help="启用后可以使用历史数据进行回放测试，结果可重复")

    # 检测分析模式是否改变
    analysis_mode_changed = st.session_state.prev_analysis_mode != analysis_mode
    st.session_state.prev_analysis_mode = analysis_mode

    # 如果分析模式改变，清除相关的会话状态
    if analysis_mode_changed:
        if 'single_results' in st.session_state:
            del st.session_state.single_results
        if 'batch_results' in st.session_state:
            del st.session_state.batch_results

    # 显示标题
    st.markdown('<h1 class="main-header">📈 智能股价预测分析系统  </h1>', unsafe_allow_html=True)
    #st.info("🔍 系统使用前复权数据，避免除权除息导致的断崖式价格变化")

    # 智能获取上次使用的开始日期
    last_start_date = CacheManager.get_last_start_date()
    if last_start_date:
        try:
            # 将字符串日期转换为datetime对象
            last_start_date_dt = datetime.strptime(last_start_date, "%Y%m%d")
        except:
            # 如果转换失败，使用默认值
            last_start_date_dt = datetime.now() - timedelta(days=3 * 365)
    else:
        # 如果没有保存的日期，使用默认值
        last_start_date_dt = datetime.now() - timedelta(days=3 * 365)

    # 数据日期设置
    start_date = st.sidebar.date_input(
        "选择数据开始日期",
        value=last_start_date_dt,
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now()
    ).strftime("%Y%m%d")

    # 回放模式特有参数
    if replay_mode:
        end_date = st.sidebar.date_input(
            "选择数据结束日期（回放模式）",
            value=datetime.now() - timedelta(days=30),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.now()
        ).strftime("%Y%m%d")

        test_start_date = st.sidebar.date_input(
            "选择测试开始日期",
            value=datetime.now() - timedelta(days=180),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.now()
        ).strftime("%Y%m%d")

        fixed_seed = st.sidebar.number_input("随机种子", min_value=1, max_value=1000, value=42)
    else:
        end_date = datetime.now().strftime("%Y%m%d")
        test_start_date = "20240101"
        fixed_seed = 42

    # 保存当前选择的开始日期
    CacheManager.save_last_start_date(start_date)

    # 交易参数
    transaction_cost = st.sidebar.slider("交易成本 (%)", 0.0, 0.5, 0.1) / 100
    slippage = st.sidebar.slider("滑点 (%)", 0.0, 0.5, 0.2) / 100

    # 集成学习设置
    st.sidebar.markdown("---")
    #st.sidebar.subheader("🎯 集成学习设置")

    use_ensemble = st.sidebar.checkbox("启用集成学习", value=True,
                                       help="使用集成学习提升模型稳定性和准确性")

    ensemble_type = st.sidebar.selectbox(
        "集成方法",
        ["voting", "stacking"],
        index=0,
        help="投票集成更快，堆叠集成更准确但更慢"
    )

    # Numba加速设置
    st.sidebar.markdown("---")
    #st.sidebar.subheader("🚀 性能加速设置")

    use_numba = st.sidebar.checkbox("启用Numba加速", value=NUMBA_AVAILABLE,
                                    disabled=not NUMBA_AVAILABLE,
                                    help="使用Numba JIT编译加速技术指标计算")

    if not NUMBA_AVAILABLE:
        st.sidebar.warning("Numba未安装，请运行: pip install numba")

    # 缓存管理
    st.sidebar.markdown("---")
    use_cache = st.sidebar.checkbox("启用数据缓存", value=True,
                                    help="启用后，股票数据将缓存到本地，提高后续访问速度")

    # 缓存文件信息
    cache_info = CacheManager.get_cache_info()
    st.sidebar.info(f"当前缓存文件数: {cache_info['file_count']}")

    if st.sidebar.button("清空缓存"):
        if CacheManager.clear_cache():
            st.sidebar.success("缓存已清空")
            st.rerun()
        else:
            st.sidebar.error("清空缓存失败")

    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### 使用说明
    - **前复权数据**: 使用前复权数据避免除权除息导致的断崖式价格变化
    - **集成学习**: 使用多个模型的集体智慧，提升预测准确性
    - **Numba加速**: 使用JIT编译加速技术指标计算，提升计算性能
    - **回放模式**: 使用历史数据进行可重复测试，确保结果一致性
    - **智能缓存**: 自动记住上次使用的开始日期，提高使用便利性
    - **优化算法**: 使用稳健的特征工程和防未来函数设计
    - **真实交易成本**: 考虑交易成本和滑点的影响
    """)

    # 显示当前模式信息
    st.sidebar.success("📊 使用前复权数据 - 避免价格断崖")

    if use_ensemble:
        st.sidebar.success(f"🤖 集成学习已启用 - {ensemble_type}")
    else:
        st.sidebar.info("🔍 使用单个模型")

    if use_numba and NUMBA_AVAILABLE:
        st.sidebar.success("🚀 Numba加速已启用")
    else:
        st.sidebar.info("🐢 Numba加速未启用")

    if replay_mode:
        st.sidebar.success("🎯 回放模式已启用")
        st.sidebar.info(f"数据范围: {start_date} 到 {end_date}")
        st.sidebar.info(f"测试开始: {test_start_date}")
        st.sidebar.info(f"随机种子: {fixed_seed}")
    else:
        st.sidebar.info("🔍 实时模式: 使用最新数据")

    # 将配置传递给分析函数
    if analysis_mode == "单支分析":
        single_stock_analysis(start_date, end_date, test_start_date, fixed_seed,
                              transaction_cost, slippage, use_cache, replay_mode,
                              use_ensemble, ensemble_type, use_numba)
    else:
        batch_analysis(start_date, end_date, test_start_date, fixed_seed,
                       transaction_cost, slippage, use_cache, replay_mode,
                       use_ensemble, ensemble_type, use_numba)


def single_stock_analysis(start_date, end_date, test_start_date, fixed_seed,
                          transaction_cost, slippage, use_cache=True, replay_mode=False,
                          use_ensemble=True, ensemble_type='voting', use_numba=True):
    """单支分析 - 支持集成学习和Numba加速"""
    st.header("🔍 单支详细分析" +
              (" (回放模式)" if replay_mode else "") +
              (" 🤖" if use_ensemble else "") +
              (" 🚀" if use_numba else "") +
              " 📊")

    col1, col2 = st.columns([1, 3])

    with col1:
        stock_symbol = st.text_input("股票代码", "600056", key="single_stock_symbol")

        # 模式信息显示
        if replay_mode:
            st.info(f"- 回放模式: {start_date} 到 {end_date}\n- 测试开始: {test_start_date}\n- 随机种子: {fixed_seed}")

        analyze_button = st.button("开始分析", type="primary", key="single_analyze_button")

        if analyze_button:
            if not stock_symbol:
                st.error("请输入股票代码")
                return

            with st.spinner("正在使用前复权数据、集成学习和Numba加速分析股票数据，请稍候..."):
                # 使用增强的预测器
                predictor = EnhancedMarketReplayPredictor(
                    symbol=stock_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    test_start_date=test_start_date,
                    fixed_seed=fixed_seed,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                    use_ensemble=use_ensemble,
                    use_numba=use_numba
                )

                results = predictor.run_enhanced_replay_analysis(
                    use_cache=use_cache,
                    ensemble_type=ensemble_type
                )

                if results:
                    st.session_state.single_results = results
                    ensemble_type_used = results.get('ensemble_type', 'individual')
                    #st.success(f"分析完成! 使用方式: {ensemble_type_used}")
                else:
                    st.error("分析失败，请检查股票代码或稍后重试")

    # 检查是否有单支分析的结果
    if 'single_results' in st.session_state:
        results = st.session_state.single_results

        # 检查结果是否有效
        if results is None or results.get('strategy_results') is None:
            st.error("策略回测失败，无法显示详细结果")
            return

        strategy = results['strategy_results']
        symbol = results['symbol']
        stock_name = results.get('stock_name', f"股票{symbol}")

        # 显示模式信息
        #st.success("📊 使用前复权数据 - 避免除权除息断崖")

        if results.get('using_ensemble', False):
            ensemble_badge = f"<span class='ensemble-badge'>集成学习 - {results.get('ensemble_type', 'voting')}</span>"
            #st.markdown(f"🎯 **分析模式**: {ensemble_badge}", unsafe_allow_html=True)
        else:
            st.info("🔍 **分析模式**: 单个模型")

        if results.get('using_numba', False):
            numba_badge = "<span class='numba-badge'>Numba加速</span>"
            #st.markdown(f"🚀 **性能加速**: {numba_badge}", unsafe_allow_html=True)
            st.info(f"特征工程耗时: {results.get('feature_time', 0):.2f}秒")
        else:
            st.info("🐢 **性能加速**: 未启用")

        if replay_mode:
            st.success(f"🎯 回放模式分析完成 (随机种子: {results.get('fixed_seed', 'N/A')})")

        # 使用增强的结果展示
        display_enhanced_results(results)

        # 交易信号详情
        st.subheader("💡 交易信号详情")

        buy_dates = strategy.get('buy_dates', [])
        sell_dates = strategy.get('sell_dates', [])

        col1, col2 = st.columns(2)

        with col1:
            st.write("**最近买入信号**")
            if len(buy_dates) > 0:
                recent_buys = strategy['df_strategy'][strategy['df_strategy']['buy_signal'] == True].tail(5)
                for idx, (date, row) in enumerate(recent_buys.iloc[::-1].iterrows()):
                    prob_percent = row['probability'] * 100
                    st.write(
                        f"{idx + 1}. {date.strftime('%Y-%m-%d')} - 价格: {row['收盘']:.2f} - 概率: {prob_percent:.2f}%")
            else:
                st.write("暂无买入信号")

        with col2:
            st.write("**最近卖出信号**")
            if len(sell_dates) > 0:
                recent_sells = strategy['df_strategy'][strategy['df_strategy']['sell_signal'] == True].tail(5)
                for idx, (date, row) in enumerate(recent_sells.iloc[::-1].iterrows()):
                    prob_percent = row['probability'] * 100
                    st.write(
                        f"{idx + 1}. {date.strftime('%Y-%m-%d')} - 价格: {row['收盘']:.2f} - 概率: {prob_percent:.2f}%")
            else:
                st.write("暂无卖出信号")

        # 交易记录表格
        st.subheader("📋 交易记录明细")

        # 获取交易记录
        trades = strategy.get('trades', [])
        if trades:
            # 创建交易记录表格
            trade_data = []
            cumulative_profit = 0.0

            for i, trade in enumerate(trades, 1):
                entry_date = trade.get('entry_date', 'N/A')
                exit_date = trade.get('exit_date', 'N/A')
                days_held = trade.get('days_held', 0)
                return_rate = trade.get('return', 0) * 100

                # 计算累计盈利
                cumulative_profit += return_rate

                # 格式化日期
                if hasattr(entry_date, 'strftime'):
                    entry_date_str = entry_date.strftime('%Y-%m-%d')
                else:
                    entry_date_str = str(entry_date)

                if hasattr(exit_date, 'strftime'):
                    exit_date_str = exit_date.strftime('%Y-%m-%d')
                else:
                    exit_date_str = str(exit_date)

                trade_data.append({
                    '序号': i,
                    '入场日期': entry_date_str,
                    '出场日期': exit_date_str,
                    '持仓天数': f"{days_held}天",
                    '收益率': f"{return_rate:.2f}%",
                    '累计盈利': f"{cumulative_profit:.2f}%"
                })

            # 创建DataFrame
            trades_df = pd.DataFrame(trade_data)
            trades_df.set_index('序号', inplace=True)
            st.dataframe(trades_df, width='stretch')

        else:
            st.info("暂无交易记录")

        # 模型性能比较
        st.subheader("📊 模型性能分析")
        fig = plot_results(results['all_results'], results.get('y_test', []),
                           symbol, stock_name, results.get('cv_results'))
        if fig:
            st.pyplot(fig)
        else:
            st.warning("无法生成模型性能图表")

        # 策略回测结果
        st.subheader("📈 策略回测表现")

        fig1 = plot_strategy_performance(
            strategy['df_strategy'], symbol, stock_name,
            buy_dates, sell_dates,
            strategy['metrics']['market_return'], strategy['metrics']['total_return']
        )
        st.pyplot(fig1)

        fig2 = plot_price_with_signals(
            strategy['df_strategy'], symbol, stock_name,
            buy_dates, sell_dates
        )
        st.pyplot(fig2)

        # 特征重要性
        st.subheader("🔍 特征重要性排名")
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(10)
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            y_pos = np.arange(len(top_features))
            bars = ax.barh(y_pos, top_features['importance'], color='#4ECDC4')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'], color='white')
            ax.set_xlabel('重要性', color='white')
            ax.set_title(f'{stock_name}({symbol}) - Top 10 特征重要性', fontsize=14, color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.grid(True, alpha=0.3)

            for spine in ax.spines.values():
                spine.set_color('white')

            st.pyplot(fig)

        # 数据质量报告
        if results.get('data_quality'):
            st.subheader("📋 数据质量报告")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("价格异常值", results['data_quality']['price_anomalies'])
            with col2:
                st.metric("成交量异常值", results['data_quality']['volume_anomalies'])
            with col3:
                st.metric("日期间隔数", results['data_quality']['date_gaps'])


def batch_analysis(start_date, end_date, test_start_date, fixed_seed,
                   transaction_cost, slippage, use_cache=True, replay_mode=False,
                   use_ensemble=True, ensemble_type='voting', use_numba=True):
    """批量分析多只股票 - 支持集成学习和Numba加速（修复版本）"""
    st.header("📋 批量股票分析" +
              (" (回放模式)" if replay_mode else "") +
              (" 🤖" if use_ensemble else "") +
              (" 🚀" if use_numba else "") +
              " 📊")

    stock_symbols_input = st.text_area("股票代码", "600056\n300308", key="batch_stock_symbols")

    # 模式信息显示
    st.info(f"🎯 批量分析设置:")
    st.info("- 数据来源: 前复权数据")
    if use_ensemble:
        st.info(f"- 集成学习: {ensemble_type}")
    else:
        st.info("- 单个模型")

    if use_numba:
        st.info("- Numba加速: 已启用")
    else:
        st.info("- Numba加速: 未启用")

    if replay_mode:
        st.info(f"- 回放模式: {start_date} 到 {end_date}")
        st.info(f"- 测试开始: {test_start_date}")
        st.info(f"- 随机种子: {fixed_seed}")

    if st.button("开始批量分析", type="primary", key="batch_analyze_button"):
        if not stock_symbols_input:
            st.error("请输入至少一个股票代码")
            return

        symbols = [s.strip() for s in stock_symbols_input.split('\n') if s.strip()]

        if not symbols:
            st.error("请输入有效的股票代码")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []

        for i, symbol in enumerate(symbols):
            status_text.text(f"正在分析 {symbol} ({i + 1}/{len(symbols)})")

            try:
                # 使用增强的预测器
                predictor = EnhancedMarketReplayPredictor(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    test_start_date=test_start_date,
                    fixed_seed=fixed_seed,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                    use_ensemble=use_ensemble,
                    use_numba=use_numba
                )

                result = predictor.run_enhanced_replay_analysis(
                    use_cache=use_cache,
                    ensemble_type=ensemble_type
                )

                if result and result.get('strategy_results'):
                    strategy = result['strategy_results']
                    stock_name = result.get('stock_name', f"股票{symbol}")
                    metrics = strategy['metrics']

                    # 使用修复后的指标计算
                    index_return = metrics.get('total_return', 0) * 100
                    market_return = metrics.get('market_return', 0) * 100
                    excess_return = metrics.get('excess_return', 0) * 100

                    # 获取信号信息
                    signal_info = strategy['signal_info']
                    signal_date = signal_info.get('date', 'N/A')
                    signal_price = signal_info.get('price', 0)
                    current_price = signal_info.get('current_price', 0)

                    # 计算价格变化百分比 - 基于最后一次交易价格
                    if signal_price and signal_price > 0:
                        price_change_pct = ((current_price - signal_price) / signal_price) * 100
                    else:
                        price_change_pct = 0

                    # 计算买入值、卖出值、总交易
                    buy_count = len(strategy.get('buy_dates', []))
                    sell_count = len(strategy.get('sell_dates', []))
                    total_trades = metrics.get('total_trades', 0)

                    # 计算准换率（准确率）
                    accuracy = result.get('best_accuracy', 0) * 100

                    # 计算胜率
                    win_rate = metrics.get('win_rate', 0) * 100

                    # 获取更多性能指标
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    max_drawdown = abs(metrics.get('max_drawdown', 0) * 100)
                    profit_factor = metrics.get('profit_factor', 0)
                    annualized_return = metrics.get('annualized_return', 0) * 100
                    avg_holding_days = metrics.get('avg_holding_days', 0)

                    # 按照参考格式构建结果字典（修复版本）
                    result_data = {
                        '股票代码': symbol,
                        '股票名称': stock_name,
                        '当前信号': signal_info['current_signal'],
                        '信号说明': signal_info['explanation'],
                        '信号日期': signal_date.strftime('%Y-%m-%d') if hasattr(signal_date, 'strftime') else str(
                            signal_date),
                        '信号价格': f"{signal_price:.2f}",
                        '当前价格': f"{current_price:.2f}",
                        '价格变化': f"{price_change_pct:.2f}%",
                        '指数收益': f"{index_return:.2f}%",
                        '市场收益': f"{market_return:.2f}%",
                        '超额收益': f"{excess_return:.2f}%",
                        '年化收益': f"{annualized_return:.2f}%",
                        '夏普比率': f"{sharpe_ratio:.2f}",
                        '最大回撤': f"{max_drawdown:.2f}%",
                        '胜率': f"{win_rate:.2f}%",
                        '盈亏比': f"{profit_factor:.2f}",
                        '买入值': buy_count,
                        '卖出值': sell_count,
                        '总交易': total_trades,
                        '平均持仓天数': f"{avg_holding_days:.1f}天",
                        '最佳模型': result['best_model'],
                        '准换率': f"{accuracy:.2f}%",
                        'AUC分数': f"{result['best_auc']:.3f}",
                        '数据开始日期': result['start_date'],
                        '数据结束日期': result['end_date'],
                        '分析模式': '回放模式' if replay_mode else '实时模式',
                        '集成学习': '是' if result.get('using_ensemble', False) else '否',
                        '集成方法': result.get('ensemble_type', 'N/A'),
                        'Numba加速': '是' if result.get('using_numba', False) else '否',
                        '数据来源': '前复权数据'
                    }

                    results.append(result_data)
                    st.success(f"✓ {symbol} 分析完成")
                else:
                    st.warning(f"⚠ 股票 {symbol} 分析失败，跳过")

            except Exception as e:
                st.error(f"❌ 分析股票 {symbol} 时出错: {str(e)}")
                logger.error(f"批量分析 {symbol} 失败: {e}")

            progress_bar.progress((i + 1) / len(symbols))

        if results:
            st.session_state.batch_results = results
            st.success(f"✅ 批量分析完成！共成功分析 {len(results)} 只股票")
        else:
            st.error("❌ 没有成功分析任何股票，请检查股票代码是否正确")

    # 检查是否有批量分析的结果
    if 'batch_results' in st.session_state:
        results = st.session_state.batch_results
        results_df = pd.DataFrame(results)

        st.subheader("📊 批量分析结果汇总")

        # 按照参考格式重新排列列的顺序（增强版本）
        column_order = [
            '股票代码', '股票名称', '当前信号', '信号说明', '信号日期', '信号价格',
            '当前价格', '价格变化', '指数收益', '市场收益', '超额收益', '年化收益',
            '夏普比率', '最大回撤', '胜率', '盈亏比', '买入值', '卖出值', '总交易',
            '平均持仓天数', '最佳模型', '准换率', 'AUC分数',
            '数据开始日期', '数据结束日期', '分析模式', '集成学习', '集成方法', 'Numba加速', '数据来源'
        ]

        # 确保只包含存在的列
        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_columns]

        def color_signal(val):
            if val == '买入':
                return 'color: red; font-weight: bold'
            elif val == '强烈买入':
                return 'color: red; font-weight: bold'
            elif val == '卖出':
                return 'color: cyan; font-weight: bold'
            elif val == '强烈卖出':
                return 'color: cyan; font-weight: bold'
            elif val == '持仓':
                return 'color: orange; font-weight: bold'
            else:
                return 'color: green; font-weight: bold'

        # 应用样式
        styled_df = results_df.style.applymap(color_signal, subset=['当前信号'])

        # 设置列宽
        column_config = {
            '股票代码': st.column_config.TextColumn(width="small"),
            '股票名称': st.column_config.TextColumn(width="small"),
            '当前信号': st.column_config.TextColumn(width="small"),
            '信号说明': st.column_config.TextColumn(width="medium"),
            '信号日期': st.column_config.TextColumn(width="small"),
            '信号价格': st.column_config.TextColumn(width="small"),
            '当前价格': st.column_config.TextColumn(width="small"),
            '价格变化': st.column_config.TextColumn(width="small"),
            '指数收益': st.column_config.TextColumn(width="small"),
            '夏普比率': st.column_config.TextColumn(width="small"),
            '胜率': st.column_config.TextColumn(width="small"),
        }

        st.dataframe(styled_df, width='stretch', height=400, column_config=column_config)

        # 信号统计
        st.subheader("📈 信号统计")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            buy_count = len(results_df[results_df['当前信号'].isin(['买入', '强烈买入'])])
            st.metric("建议买入", buy_count)

        with col2:
            sell_count = len(results_df[results_df['当前信号'].isin(['卖出', '强烈卖出'])])
            st.metric("建议卖出", sell_count)

        with col3:
            hold_count = len(results_df[results_df['当前信号'] == '持仓'])
            st.metric("建议持仓", hold_count)

        with col4:
            watch_count = len(results_df[results_df['当前信号'] == '观望'])
            st.metric("建议观望", watch_count)

        # 性能指标统计
        st.subheader("📊 性能指标统计")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # 计算平均年化收益
            try:
                annual_returns = results_df['年化收益'].str.rstrip('%').astype(float)
                avg_annual_return = annual_returns.mean()
                st.metric("平均年化收益", f"{avg_annual_return:.2f}%")
            except:
                st.metric("平均年化收益", "N/A")

        with col2:
            try:
                sharpe_ratios = results_df['夏普比率'].astype(float)
                avg_sharpe = sharpe_ratios.mean()
                st.metric("平均夏普比率", f"{avg_sharpe:.2f}")
            except:
                st.metric("平均夏普比率", "N/A")

        with col3:
            try:
                win_rates = results_df['胜率'].str.rstrip('%').astype(float)
                avg_win_rate = win_rates.mean()
                st.metric("平均胜率", f"{avg_win_rate:.2f}%")
            except:
                st.metric("平均胜率", "N/A")

        with col4:
            try:
                profit_factors = results_df['盈亏比'].astype(float)
                avg_profit_factor = profit_factors.mean()
                st.metric("平均盈亏比", f"{avg_profit_factor:.2f}")
            except:
                st.metric("平均盈亏比", "N/A")

        # 可视化分析
        st.subheader("📊 批量分析可视化")

        if len(results) > 0:
            col1, col2 = st.columns(2)

            with col1:
                # 信号分布饼图
                try:
                    setup_chinese_font()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')

                    signals = [r['当前信号'] for r in results]
                    signal_counts = pd.Series(signals).value_counts()

                    if len(signal_counts) > 0:
                        color_map = {
                            '强烈买入': '#4ECDC4',
                            '买入': '#4ECDC4',
                            '强烈卖出': '#FF6B6B',
                            '卖出': '#FF6B6B',
                            '持仓': '#45B7D1',
                            '观望': '#FFA07A'
                        }
                        colors = [color_map.get(sig, '#45B7D1') for sig in signal_counts.index]

                        wedges, texts, autotexts = ax.pie(
                            signal_counts.values,
                            labels=signal_counts.index,
                            autopct='%1.1f%%',
                            colors=colors,
                            startangle=90
                        )

                        for text in texts:
                            text.set_color('white')
                            text.set_fontsize(10)
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(10)

                        ax.set_title('当前信号分布', fontsize=14, color='white')
                    else:
                        ax.text(0.5, 0.5, '无信号数据', ha='center', va='center',
                                transform=ax.transAxes, fontsize=14, color='white')
                        ax.set_title('当前信号分布', fontsize=14, color='white')

                    for spine in ax.spines.values():
                        spine.set_color('white')

                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"绘制信号分布图时出错: {e}")

            with col2:
                # 年化收益分布直方图
                try:
                    setup_chinese_font()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')

                    returns = []
                    for r in results:
                        try:
                            return_str = r['年化收益'].replace('%', '')
                            returns.append(float(return_str))
                        except:
                            continue

                    if len(returns) > 0:
                        ax.hist(returns, bins=10, color='#4ECDC4', alpha=0.7, edgecolor='white')
                        ax.set_xlabel('年化收益 (%)', color='white')
                        ax.set_ylabel('股票数量', color='white')
                        ax.set_title('年化收益分布', fontsize=14, color='white')
                        ax.tick_params(axis='x', colors='white')
                        ax.tick_params(axis='y', colors='white')
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, '无有效收益数据', ha='center', va='center',
                                transform=ax.transAxes, fontsize=14, color='white')
                        ax.set_title('年化收益分布', fontsize=14, color='white')

                    for spine in ax.spines.values():
                        spine.set_color('white')

                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"绘制收益分布图时出错: {e}")

            # 模型性能比较
            try:
                st.subheader("🤖 模型性能比较")

                model_accuracies = {}
                for result in results:
                    model_name = result['最佳模型']
                    accuracy_str = result['准换率'].replace('%', '')
                    accuracy = float(accuracy_str)

                    if model_name not in model_accuracies:
                        model_accuracies[model_name] = []
                    model_accuracies[model_name].append(accuracy)

                if len(model_accuracies) > 0:
                    model_avg_accuracies = {model: np.mean(accs) for model, accs in model_accuracies.items()}

                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')

                    models = list(model_avg_accuracies.keys())
                    avg_accuracies = [model_avg_accuracies[model] for model in models]

                    bars = ax.bar(models, avg_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A'])
                    ax.set_ylabel('平均准确率 (%)', color='white')
                    ax.set_title('各模型平均准确率比较', fontsize=14, color='white')
                    ax.tick_params(axis='x', colors='white', rotation=45)
                    ax.tick_params(axis='y', colors='white')
                    ax.grid(True, alpha=0.3)

                    for bar, accuracy in zip(bars, avg_accuracies):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                f'{accuracy:.2f}%', ha='center', va='bottom',
                                color='white', fontweight='bold')

                    for spine in ax.spines.values():
                        spine.set_color('white')

                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"无法绘制模型性能比较图: {e}")

        # 性能排名
        st.subheader("🏆 股票性能排名")

        try:
            performance_df = results_df.copy()

            def parse_float(value):
                try:
                    if isinstance(value, str):
                        return float(value.replace('%', ''))
                    return float(value)
                except:
                    return 0.0

            # 创建综合评分
            performance_df['夏普数值'] = performance_df['夏普比率'].apply(parse_float)
            performance_df['年化数值'] = performance_df['年化收益'].apply(parse_float)
            performance_df['胜率数值'] = performance_df['胜率'].apply(parse_float)

            # 综合评分 = 夏普比率 * 0.4 + 年化收益 * 0.4 + 胜率 * 0.2
            performance_df['综合评分'] = (
                    performance_df['夏普数值'] * 0.4 +
                    performance_df['年化数值'] * 0.01 * 0.4 +  # 年化收益需要缩小
                    performance_df['胜率数值'] * 0.01 * 0.2  # 胜率需要缩小
            )

            performance_df = performance_df.sort_values('综合评分', ascending=False)

            st.dataframe(
                performance_df[
                    ['股票代码', '股票名称', '当前信号', '年化收益', '夏普比率', '胜率', '综合评分']].head(10),
                width='stretch')

            # 推荐关注的股票
            buy_recommendations = performance_df[performance_df['当前信号'].isin(['买入', '强烈买入'])]
            if len(buy_recommendations) > 0:
                st.subheader("💡 推荐关注股票")
                st.dataframe(
                    buy_recommendations[
                        ['股票代码', '股票名称', '当前信号', '年化收益', '夏普比率', '信号日期', '信号价格',
                         '当前价格']].head(5),
                    width='stretch')
            else:
                st.info("暂无强烈推荐的买入股票")

        except Exception as e:
            st.warning(f"无法生成性能排名: {e}")

        # 下载按钮
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载分析结果 (CSV)",
            data=csv,
            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )

        # 分析总结
        st.subheader("📋 分析总结")

        if len(results) > 0:
            total_stocks = len(results)
            avg_annual_return = annual_returns.mean() if 'annual_returns' in locals() else 0
            avg_sharpe = sharpe_ratios.mean() if 'sharpe_ratios' in locals() else 0

            st.info(f"""
            **📊 总体表现:**
            - 分析股票数量: {total_stocks} 只
            - 平均年化收益: {avg_annual_return:.2f}%
            - 平均夏普比率: {avg_sharpe:.2f}
            - 建议买入股票: {buy_count} 只
            - 建议卖出股票: {sell_count} 只
            """)


if __name__ == "__main__":
    main()
    #streamlit run C:\Users\Kaplony\PycharmProjects\PythonProject\stock_app_02.py [ARGUMENTS]