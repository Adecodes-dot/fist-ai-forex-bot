import streamlit as st
import pandas as pd
import os
from datetime import datetime

def load_env():
    # Load settings from .env file
    config = {}
    with open('.env') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k,v = line.strip().split('=',1)
                config[k] = v
    return config

def save_env(config):
    lines = []
    with open('.env','r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k,_ = line.strip().split('=',1)
                if k in config:
                    lines.append(f"{k}={config[k]}\n")
                    continue
            lines.append(line)
    with open('.env','w') as f:
        f.writelines(lines)

st.set_page_config(page_title='FIST Algo Monitor', layout='wide')
st.title('FIST Trading Algorithm Dashboard')

# Sidebar: Settings
st.sidebar.header('Configuration')
env = load_env()
MODE = st.sidebar.selectbox('Mode', ['train','live'], index=['train','live'].index(env.get('MODE','live')))
SYMBOL = st.sidebar.text_input('Symbol', env.get('SYMBOL','EURUSD'))
CONF_THRESHOLD = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, float(env.get('CONF_THRESHOLD',0.7)), step=0.01)
RISK_PCT = st.sidebar.slider('Risk % per Trade', 0.0, 0.1, float(env.get('RISK_PCT',0.01)), step=0.005)

if st.sidebar.button('Save Configuration'):
    env['MODE'] = MODE
    env['SYMBOL'] = SYMBOL
    env['CONF_THRESHOLD'] = str(CONF_THRESHOLD)
    env['RISK_PCT'] = str(RISK_PCT)
    save_env(env)
    st.sidebar.success('Configuration saved! Restart your bot to apply.')

# Main: Live performance
st.header('Live Trade Log')
if os.path.exists('fp_trades.csv'):
    df = pd.read_csv('fp_trades.csv', header=None,
                     names=['timestamp','symbol','side','price','sl','tp','pnl','balance'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    st.dataframe(df.sort_values('timestamp', ascending=False).head(50))
    # Metrics
    st.subheader('Metrics')
    last_balance = df['balance'].iloc[-1]
    start_balance = df['balance'].iloc[0]
    dd = (df['balance'].cummax() - df['balance']) / df['balance'].cummax()
    max_dd = dd.max()
    st.metric('Current Balance', f"${last_balance:.2f}", delta=f"{last_balance-start_balance:.2f}")
    st.metric('Max Drawdown', f"{max_dd:.2%}")
else:
    st.info('No trades logged yet.')

st.header('Equity Curve')
if os.path.exists('equity_curve.csv'):
    eq = pd.read_csv('equity_curve.csv', names=['timestamp','equity'])
    eq['timestamp']=pd.to_datetime(eq['timestamp'])
    st.line_chart(eq.set_index('timestamp')['equity'])
else:
    st.info('No equity snapshots yet.')

st.header('Logs')
if os.path.exists('trading_log.txt'):
    log_text = open('trading_log.txt').read().splitlines()[-50:]
    st.text('\n'.join(log_text))
else:
    st.info('No logs found.')

st.markdown('---')
st.write('Last updated:', datetime.utcnow())
