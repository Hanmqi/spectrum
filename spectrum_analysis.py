# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 2023

@author: 中国科学院广州地球化学研究所韩梦麒
@designer involved: 中国科学院广州地球化学研究所张天琦、陈可妍、秦效荣
"""


import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import os
import pysptools
import scipy.signal as signal
from pysptools import spectro
from specdal import Collection, Spectrum


@st.cache_data
def readspectrum(path):
    # 对asd格式的近红外光谱进行读取并预处理,输出dataframe

    datadir = path
    # 设置Collection类
    c = Collection(name='spectrum data')
    for f in sorted(os.listdir(datadir)):
        if f.split('.')[-1] == 'asd':
            spectrum = Spectrum(filepath=os.path.join(datadir, f))
            c.append(spectrum)

    # 断点矫正，暂时弃用，改用自写程序。目前没有理解这个的具体算法
    # c.jump_correct(splices=[1000, 1800], reference=0)
    df = jump_correct(c.data.T)

    return df


@st.cache_data
def continuum_removal(dataframe):
    # 对输入的dataframe格式的光谱进行包络线去除并以dataframe的形式输出
    spectrum_data = dataframe.values
    spectrum_names = list(dataframe.index.values)
    wavelength = np.array(dataframe.columns.values)
    spectrum_data_removals = []
    for index in range(0, len(spectrum_data)):
        # 去除包络线
        spectrum_data_removal = pysptools.spectro.convex_hull_removal(
            spectrum_data[index], wavelength)
        spectrum_data_removals.append(spectrum_data_removal[0])
    dataframe_removal = pd.DataFrame(
        index=spectrum_names, columns=wavelength, data=np.array(spectrum_data_removals))

    return dataframe_removal


@st.cache_data
def input_asd(path):
    df = readspectrum(path)
    return df


@st.cache_data
def input_csv(path):
    # 读取csv文件，未写完
    df = pd.read_csv(path)
    # 判断csv格式
    df_return = pd.DataFrame()
    if isinstance(df.iloc[0, 0], str):
        # df.iloc[0, 0]为str，代表未将名称读入index
        index = df.iloc[:, 0].values[:]
        df_return = df.iloc[:, 1:]
        coloumns = df_return.columns.values
        df_return.columns = [float(x) for x in coloumns]
        df_return.index = index
    elif isinstance(df.iloc[0, 0], float):
        # df.iloc[0, 0]为float，代表df.values为数据
        df_return = df
        coloumns = df_return.columns.values[1:]
        df_return.columns = [float(x) for x in coloumns]
    return df_return


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


@st.cache_data
def spectrum_diff(df, number):
    return df.T.diff(periods=number).T


def plot(df):
    # 根据选项绘制光谱图
    col_list = st.session_state.df.index
    fig = px.line(df.T, x=st.session_state.df.columns, y=col_list)
    # 使标签放置在图片底部
    # fig.update_layout(legend=dict(orientation='h'))
    fig.update_layout(xaxis=dict(title=dict(text='wavelength')))
    st.plotly_chart(fig, themes='streamlit')


@st.cache_data
def split_waveband(waveband_str):
    # 分割输入的光谱波段
    waveband = waveband_str.split(' ')
    waveband_data = np.zeros((len(waveband), 2))
    for index in range(len(waveband)):
        waveband_data[index] = waveband[index].split('-')
    return waveband_data


@st.cache_data
def find_peaks_index(df, waveband):
    # 根据输入的波段和光谱寻找对应波段的极值点的波长
    df_data = df.values
    peaks_data = np.zeros((len(df_data), len(waveband)))
    for index in range(len(df_data)):
        num_peak = signal.find_peaks(df_data[index]*-1, distance=10)
        peaks = num_peak[0][0:] + 350
        for waveband_index in range(len(waveband)):
            temp_1 = peaks[peaks > (waveband[waveband_index][0])]
            temp_2 = temp_1[temp_1 <= (waveband[waveband_index][1])]
            if temp_2.size > 0:
                peaks_data[index][waveband_index] = temp_2[0]
            else:
                peaks_data[index][waveband_index] = np.NaN
    return peaks_data


@st.cache_data
def find_peaks_values(df, peaks_data):
    # 根据极值点的波长找反射率
    df_data = df.values
    peaks_and_reflectance_data = np.zeros((len(peaks_data), len(peaks_data[0]) * 2))
    for sample_index in range(len(peaks_data)):
        for waveband_index in range(len(peaks_data[0])):
            wavelength = peaks_data[sample_index][waveband_index]
            peaks_and_reflectance_data[sample_index][waveband_index * 2] = wavelength
            if not np.isnan(peaks_data[sample_index][waveband_index]):
                peaks_and_reflectance_data[sample_index][waveband_index * 2 + 1] = df_data[sample_index][int(wavelength - 350)]
            else:
                peaks_and_reflectance_data[sample_index][waveband_index * 2 + 1] = None

    return peaks_and_reflectance_data


@st.cache_data
def find_peaks(df, wavebandinput):
    # 根据输入的波段和光谱寻找对应波段的极值点的所有特征
    waveband = split_waveband(wavebandinput)
    peaks_data = find_peaks_index(df, waveband)
    find_peaks_data = find_peaks_values(df, peaks_data)
    columns = wavebandinput.split(' ')
    for column in range(len(columns)):
        columns.insert(column*2+1, columns[column*2] + 'Values')
    find_peaks_df = pd.DataFrame(data=find_peaks_data, index=df.index, columns=columns)
    return find_peaks_df


@st.cache_data
def single_breakpoint_correct(single_data, position):
    # 断点矫正程序3
    single_data_correct = single_data.copy()
    position -= 350
    difference = (single_data_correct[position+1] - single_data_correct[position]) - \
                 (single_data_correct[position+2] - single_data_correct[position+1])
    for column in range(position+1, len(single_data_correct)):
        single_data_correct[column] -= difference
    return single_data_correct


@st.cache_data
def all_breakpoint_correct(data, position):
    # 断点矫正程序2
    data_correct = data.copy()
    for raw in range(0, len(data)):
        data_correct[raw, :] = single_breakpoint_correct(data[raw, :], position)
    return data_correct


@st.cache_data
def jump_correct(df):
    # 断点矫正程序1
    df_new = df.copy()
    data = df.values
    data = all_breakpoint_correct(data, 1000)
    data = all_breakpoint_correct(data, 1800)
    df_new.values[:, :] = data
    return df_new


st.set_page_config(page_title='GIG光谱分析', page_icon='🛰', layout='centered', initial_sidebar_state='auto', menu_items=None)

# 创建全局变量
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if 'df_er' not in st.session_state:
    st.session_state.df_er = pd.DataFrame()

if 'df_diff1' not in st.session_state:
    st.session_state.df_diff1 = pd.DataFrame()

if 'df_diff2' not in st.session_state:
    st.session_state.df_diff2 = pd.DataFrame()

if 'name' not in st.session_state:
    st.session_state.name = str()

if 'df_dict' not in st.session_state:
    st.session_state.df_dict = {}

# 创建侧边栏
st.sidebar.title('ASD光谱处理')
st.sidebar.subheader('导入数据')
path = st.sidebar.text_input('输入文件夹路径如"E:\\硕士\\梅州样品\\unbaked\\RJ-D2-unbaked"')
st.session_state.name = path.split('\\')[-1]
if path != '':
    # 提取数据
    # 判断输入的路径是文件夹还是csv
    if path[-4:] == '.csv':
        st.session_state.df = input_csv(path)
    else:
        st.session_state.df = input_asd(path)
    st.session_state.df_er = continuum_removal(st.session_state.df)
    st.session_state.df_diff1 = spectrum_diff(st.session_state.df, 1)
    st.session_state.df_diff2 = spectrum_diff(st.session_state.df, 2)
    st.session_state.df_dict = {'原光谱': st.session_state.df,
                                '一阶导': st.session_state.df_diff1,
                                '二阶导': st.session_state.df_diff2,
                                '去包络线光谱': st.session_state.df_er,
                                }
if len(st.session_state.df):
    def callback():
        pass
    st.sidebar.subheader('绘制光谱')
    show_radio = st.sidebar.radio(label='PICK ONE',
                                  options=("原光谱", "一阶导", "二阶导", "去包络线光谱"),  # 选项
                                  index=0,  # 初始化选项
                                  format_func=lambda x: f"{x}",  # 这是格式化函数，注意我把原始选项ABC修改了。一般地，跟字典或dataframe结合也很好用。
                                  key="radio_demo",
                                  help='HMQ❤LLY',  # 看到组件右上角的问号了没？上去悬停一下。
                                  on_change=callback, args=None, kwargs=None)
    st.header('光谱图')
    # 寻找极值点模块
    st.sidebar.subheader('寻找极值点')
    wavebandinput = st.sidebar.text_input('输入需要的波段如"1380-1395 1400-1420 1900-1920"')
    # 渲染数据
    if show_radio:
        # st.subheader(show_radio)
        plot(st.session_state.df_dict[show_radio])
        # 下载光谱数据
        csv = convert_df(st.session_state.df_dict[show_radio])
        st.download_button(label='下载数据',
                           data=csv,
                           file_name=st.session_state.name + show_radio + '数据.csv',
                           mime='text/csv')
        if wavebandinput != '':
            peaks_df = find_peaks(st.session_state.df_dict[show_radio], wavebandinput)
            st.header('极值点')
            st.dataframe(peaks_df)
            csv = convert_df(peaks_df)
            st.download_button(label='下载极值点数据',
                               data=csv,
                               file_name=st.session_state.name + '极值点.csv',
                               mime='text/csv')

st.sidebar.info('📬联系我们: blkmoo@163.com')
