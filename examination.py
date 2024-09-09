# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:30:16 2024

@author: AD1501015P01
"""

import streamlit as st
import pandas as pd
import plotly.express as px


# 데이터 파일 경로
file_path_sku = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/1.진행SKU.xlsx'
file_path_discontinued_sku = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/2.단종SKU.xlsx'
file_path_sales = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/3.소분류매출.xlsx'
file_path_calculate = "C:/Users/AD1501015P01/Desktop/대시보드/streamlit/4.소싱별SKU당판매수량(진행).xlsx"
file_path_inventory = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/5.진행재고액.xlsx'
file_path_inventory2 = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/6.단종재고액.xlsx'
file_path_sales_range = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/7.판매구간별SKU수.xlsx'
file_path_aging_inventory = 'C:/Users/AD1501015P01/Desktop/대시보드/streamlit/8.월령별재고액.xlsx'
file_path_order_amt = "C:/Users/AD1501015P01/Desktop/대시보드/streamlit/9.상품발주.xlsx"
file_path_best_worst = "C:/Users/AD1501015P01/Desktop/대시보드/streamlit/10.상품별판매.xlsx"


@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# 데이터 로드
df_sku = load_data(file_path_sku)
df_discontinued_sku = load_data(file_path_discontinued_sku)
df_sales = load_data(file_path_sales)
df_inventory = load_data(file_path_inventory)
df_discontinued_inventory = load_data(file_path_inventory2)
df_sales_range = load_data(file_path_sales_range)
df_aging_inventory = load_data(file_path_aging_inventory)
df_forwarding_calculate = load_data(file_path_calculate)
df_order_amt = load_data(file_path_order_amt)
df_best_worst = load_data(file_path_best_worst)


# NaN을 0으로 대체
df_inventory.fillna(0, inplace=True)
df_discontinued_inventory.fillna(0, inplace=True)
df_sales_range.fillna(0, inplace=True)
df_aging_inventory.fillna(0, inplace=True)
df_forwarding_calculate.fillna(0, inplace=True)
df_order_amt.fillna(0,inplace=True)
df_best_worst.fillna(0, inplace=True)


# 사이드바 설정
st.sidebar.markdown("""
    <div style="text-align: center;">
        <h1>신상품 발주검토 대시보드</h1>
        <div style="text-align: right;">
            <p> Made by 데이터분석팀</p>
            <p> </p>
            <p> </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
subcategory_input = st.sidebar.text_input('카테고리 소분류 입력')
subcategories = df_sku['카테고리소분류'].unique()
subcategory_select = st.sidebar.selectbox('카테고리 소분류 선택', [''] + list(subcategories))
selected_subcategory = subcategory_input if subcategory_input else subcategory_select

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
total_order_quantity = st.sidebar.number_input("초도발주 예정수량", min_value=1, step=1)
import_type = st.sidebar.selectbox("수입 구분", ["국내", "수입"])


# 데이터 전처리
df_forwarding_calculate.columns = df_forwarding_calculate.columns.str.strip()  # 열 이름의 공백 제거
df_forwarding_calculate = df_forwarding_calculate.rename(columns=lambda x: x.split('년')[0] + '년' if '년' in x else x)
df_forwarding_calculate_melted = df_forwarding_calculate.melt(id_vars=['카테고리소분류', '수입구분'], var_name='연도', value_name='월평균판매수량')


# 데이터의 '연도'별로 월평균 판매량 계산
filtered_data = df_forwarding_calculate_melted[
    (df_forwarding_calculate_melted['카테고리소분류'] == selected_subcategory) &
    (df_forwarding_calculate_melted['수입구분'] == import_type)
]

# 연도별 평균 판매 수량 계산 및 소요 개월 수 계산

for year in filtered_data['연도'].unique():
    year_data = filtered_data[filtered_data['연도'] == year]
    avg_sales = year_data['월평균판매수량'].mean()
    months_needed = total_order_quantity / avg_sales if avg_sales > 0 else float('inf')
    
    avg_sales_formatted = f"{avg_sales:,.0f}"
    total_order_quantity_formatted = f"{total_order_quantity:,}"
    months_needed_formatted = f"{months_needed:.1f}"
    
    st.sidebar.write(f"■ {year} 월평균 판매수량: {avg_sales_formatted}개")
    st.sidebar.write(f"■ 입력 초도발주량 소요 : {months_needed_formatted}개월")
  


# CSS 스타일 추가
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 22px; /* 폰트 크기 조정 */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px; /* 폰트 크기 조정 */
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px; /* 폰트 크기 조정 */
    }
    </style>
""", unsafe_allow_html=True)

if selected_subcategory:
    try:
        # 항목 1: 현재와 전월의 진행 SKU (매번교체)
        
        
        
        current_sku = df_sku[df_sku['카테고리소분류'] == selected_subcategory]['2024년09월']
        prev_sku = df_sku[df_sku['카테고리소분류'] == selected_subcategory]['2024년08월']
        
        if not current_sku.empty and not prev_sku.empty:
            current_sku = current_sku.values[0]
            prev_sku = prev_sku.values[0]
            sku_change = int(current_sku - prev_sku)
        else:
            st.write("진행 SKU 데이터가 없습니다.")
            current_sku, prev_sku, sku_change = 0, 0, 0

        # 진행 SKU 순위 계산
        df_sku['순위'] = df_sku['2024년09월'].rank(method='min', ascending=False)
        df_sku['전월 순위'] = df_sku['2024년08월'].rank(method='min', ascending=False)
        
        if not df_sku[df_sku['카테고리소분류'] == selected_subcategory].empty:
            current_rank = df_sku[df_sku['카테고리소분류'] == selected_subcategory]['순위'].values[0]
            prev_rank = df_sku[df_sku['카테고리소분류'] == selected_subcategory]['전월 순위'].values[0]
            rank_change = int(prev_rank - current_rank)
        else:
            st.write("진행 SKU 순위 데이터가 없습니다.")
            current_rank, prev_rank, rank_change = 0, 0, 0
        
        total_count = len(df_sku)

        # 항목 3: 현재 단종 SKU
        
        

        current_discontinued_sku = df_discontinued_sku[df_discontinued_sku['카테고리소분류'] == selected_subcategory]['2024년09월']
        prev_discontinued_sku = df_discontinued_sku[df_discontinued_sku['카테고리소분류'] == selected_subcategory]['2024년08월']
        
        if not current_discontinued_sku.empty and not prev_discontinued_sku.empty:
            current_discontinued_sku = current_discontinued_sku.values[0]
            prev_discontinued_sku = prev_discontinued_sku.values[0]
            discontinued_sku_change = int(current_discontinued_sku - prev_discontinued_sku)
        else:
            st.write("단종 SKU 데이터가 없습니다.")
            current_discontinued_sku, prev_discontinued_sku, discontinued_sku_change = 0, 0, 0

        # 항목 4: 단종 SKU 순위
        df_discontinued_sku['순위'] = df_discontinued_sku['2024년09월'].rank(method='min', ascending=False)
        df_discontinued_sku['전월 순위'] = df_discontinued_sku['2024년08월'].rank(method='min', ascending=False)
        
        if not df_discontinued_sku[df_discontinued_sku['카테고리소분류'] == selected_subcategory].empty:
            discontinued_rank = df_discontinued_sku[df_discontinued_sku['카테고리소분류'] == selected_subcategory]['순위'].values[0]
            prev_discontinued_rank = df_discontinued_sku[df_discontinued_sku['카테고리소분류'] == selected_subcategory]['전월 순위'].values[0]
            discontinued_rank_change = int(prev_discontinued_rank - discontinued_rank)
        else:
            st.write("단종 SKU 순위 데이터가 없습니다.")
            discontinued_rank, prev_discontinued_rank, discontinued_rank_change = 0, 0, 0

        # 항목 5: 진행 재고액  
        
        
        
        # 항목 1: 현재와 전월의 진행 SKU (매번교체)
        
        
        df_inventory['순위'] = df_inventory['2024년09월'].rank(method='min', ascending=False)
        df_inventory['전월 순위'] = df_inventory['2024년08월'].rank(method='min', ascending=False)
        
        if not df_inventory[df_inventory['카테고리소분류'] == selected_subcategory].empty:
            current_progress_inventory = df_inventory[df_inventory['카테고리소분류'] == selected_subcategory]['2024년09월'].values[0] / 1e8
            prev_progress_inventory = df_inventory[df_inventory['카테고리소분류'] == selected_subcategory]['2024년08월'].values[0] / 1e8
            progress_inventory_change = current_progress_inventory - prev_progress_inventory

            progress_rank = df_inventory[df_inventory['카테고리소분류'] == selected_subcategory]['순위'].values[0]
            prev_progress_rank = df_inventory[df_inventory['카테고리소분류'] == selected_subcategory]['전월 순위'].values[0]
            progress_rank_change = int(prev_progress_rank - progress_rank)
        else:
            st.write("진행 재고액 데이터가 없습니다.")
            current_progress_inventory, prev_progress_inventory, progress_inventory_change, progress_rank, progress_rank_change = 0, 0, 0, 0, 0

        # 항목 6: 단종 재고액
        df_discontinued_inventory['순위'] = df_discontinued_inventory['2024년09월'].rank(method='min', ascending=False)
        df_discontinued_inventory['전월 순위'] = df_discontinued_inventory['2024년08월'].rank(method='min', ascending=False)
        
        if not df_discontinued_inventory[df_discontinued_inventory['카테고리소분류'] == selected_subcategory].empty:
            current_discontinued_inventory = df_discontinued_inventory[df_discontinued_inventory['카테고리소분류'] == selected_subcategory]['2024년09월'].values[0] / 1e8
            prev_discontinued_inventory = df_discontinued_inventory[df_discontinued_inventory['카테고리소분류'] == selected_subcategory]['2024년08월'].values[0] / 1e8
            discontinued_inventory_change = current_discontinued_inventory - prev_discontinued_inventory

            discontinued_rank = df_discontinued_inventory[df_discontinued_inventory['카테고리소분류'] == selected_subcategory]['순위'].values[0]
            prev_discontinued_rank = df_discontinued_inventory[df_discontinued_inventory['카테고리소분류'] == selected_subcategory]['전월 순위'].values[0]
            discontinued_rank_change = int(prev_discontinued_rank - discontinued_rank)
        else:
            current_discontinued_inventory, prev_discontinued_inventory, discontinued_inventory_change, discontinued_rank, discontinued_rank_change = 0, 0, 0, 0, 0

        # 총 재고 개수
        total_inventory_count = len(df_inventory)
        total_discontinued_count = len(df_discontinued_inventory)


        st.markdown("<h2>진행/단종 현황</h2>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
             st.metric(label="진행 SKU", value=f"{int(current_sku):,}개", delta=f"{sku_change:,}개")
        with col2:
             st.metric(label="진행 재고액", value=f"{current_progress_inventory:.2f}억", delta=f"{progress_inventory_change:.2f}억")
        with col3:
             st.metric(label="진행 SKU 순위", value=f"{int(current_rank)}위 / {total_count}위", delta=f"{rank_change}위")
        with col4:
             st.metric(label="진행 재고액 순위", value=f"{int(progress_rank)}위 / {total_inventory_count}위", delta=f"{progress_rank_change}위")
             st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric(label="단종 SKU", value=f"{int(current_discontinued_sku):,}개", delta=f"{discontinued_sku_change:,}개")
        with col6:
            st.metric(label="단종 재고액", value=f"{current_discontinued_inventory:.2f}억", delta=f"{discontinued_inventory_change:.2f}억")
        with col7:
            st.metric(label="단종 SKU 순위", value=f"{int(discontinued_rank)}위 / {total_discontinued_count}위", delta=f"{discontinued_rank_change}위")
        with col8:
            st.metric(label="단종 재고액 순위", value=f"{int(discontinued_rank)}위 / {total_discontinued_count}위", delta=f"{discontinued_rank_change}위")
        st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # 항목 5 및 항목 6: 소분류별 매출
        df_sales['2024년 월평균 매출'] = df_sales[[f'2024년{str(i).zfill(2)}월' for i in range(1, 10)]].mean(axis=1) / 1e8
        df_sales_ranked = df_sales.sort_values(by='2024년 월평균 매출', ascending=False).reset_index(drop=True)
        rank = df_sales_ranked[df_sales_ranked['카테고리소분류'] == selected_subcategory].index[0] + 1
        total_sales_count = len(df_sales_ranked)


        df_forwarding_calculate.columns = df_forwarding_calculate.columns.str.strip()  # Remove any whitespace in column names
        df_forwarding_calculate = df_forwarding_calculate.rename(columns=lambda x: x.split('년')[0] + '년' if '년' in x else x)
        df_forwarding_calculate_melted = df_forwarding_calculate.melt(id_vars=['카테고리소분류', '수입구분'], var_name='연도', value_name='월평균판매수량')

        df_sales_filtered = df_sales[df_sales['카테고리소분류'] == selected_subcategory]

        if not df_sales_filtered.empty:
            sales_2024 = df_sales_filtered[[f'2024년{str(i).zfill(2)}월' for i in range(1, 10)]].mean(axis=1).values[0] / 1e8
            sales_2023 = df_sales_filtered[[f'2023년{str(i).zfill(2)}월' for i in range(1, 13)]].mean(axis=1).values[0] / 1e8
            sales_change = sales_2024 - sales_2023

            avg_sales_quantity_2024 = df_forwarding_calculate_melted[
                (df_forwarding_calculate_melted['카테고리소분류'] == selected_subcategory) &
                (df_forwarding_calculate_melted['연도'] == '2024년')
            ]['월평균판매수량'].mean()

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<h2>소분류별 매출</h2>", unsafe_allow_html=True)
            st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
 
            col15, col16, col17 = st.columns(3)

            with col15:
                st.metric(label="24년 평균판매수량", value=f"{avg_sales_quantity_2024:,.0f}개")
                
            with col16:
                st.metric(label="24년 월평균 매출", value=f"{sales_2024:.2f}억", delta=f"{sales_change:.2f}억")

            with col17:
                st.metric(label="매출 순위", value=f"{rank}위 / {total_sales_count}위")
                
        sales_columns = [f'2023년{str(i).zfill(2)}월' for i in range(1, 13)] + [f'2024년{str(i).zfill(2)}월' for i in range(1, 10)]
        df_sales_filtered = df_sales_filtered[['카테고리소분류'] + sales_columns]

        # melt를 사용하여 데이터 변형
        df_sales_melted = df_sales_filtered.melt(id_vars=['카테고리소분류'], var_name='월', value_name='매출액')

        # 억 단위로 변환된 매출액 열 추가
        df_sales_melted['매출액 (억)'] = df_sales_melted['매출액'] / 1e8

        # 그래프 생성
        fig = px.line(df_sales_melted, 
                      x='월', y='매출액 (억)', title=f"{selected_subcategory} 매출 추이", markers=True,
                      text='매출액 (억)')  # 데이터 포인트에 텍스트 추가

        # 세로 점선 추가
        fig.add_vline(x='2024년01월', line_dash="dash", line_color="grey")

        # 마커에 마우스를 올리면 억 단위로 표시되게 하기
        fig.update_traces(
            line_color='black',
            marker_symbol='circle', 
            marker_color='white', 
            marker_size=8, 
            marker_line_width=3, 
            marker_line_color='black',
            texttemplate='%{text:.1f}억',  
            hovertemplate='%{x}<br>매출액: %{y:.1f}억', 
            textfont=dict(
                color='black'  
                )
            )
        
        max_sales = df_sales_melted['매출액'].max()
        fig.update_layout(
            yaxis=dict(
                tickvals=[i*1e8 for i in range(0, int(max_sales / 1e8) + 1)],  # 눈금 값 설정
                ticktext=[f'{int(i)}억' for i in range(0, int(max_sales / 1e8) + 1)],  # 눈금 레이블 설정
                title='매출액 (억)'  # y축 제목
            ),
            xaxis_tickfont_color='black',
            yaxis_title_font_color='black',
            yaxis_tickfont_color='black',
            paper_bgcolor='white',
            plot_bgcolor='white',
            title={ 
                'text': f"{selected_subcategory} 매출 추이",
                'x': 0.5,
                'xanchor': 'center',
                'font': {   
                    'size': 20,
                    'color': 'black'
                }
            },
            autosize=True  # 화면 크기에 맞게 자동 조정
        )
    
    
        # 그래프 값 표시
        fig.update_traces(texttemplate='%{y:.1f}억', textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 항목 7: 판매구간별 SKU 수
        
        
        
        
        # 항목 1: 현재와 전월의 진행 SKU (매번교체)
        
        
        
        
        
        st.markdown("<h2>판매구간별 SKU 수</h2>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)

        df_sales_range_filtered = df_sales_range[df_sales_range['카테고리소분류'] == selected_subcategory]
        if not df_sales_range_filtered.empty:
            # 전처리
            bins = [0, 1000, 3000, 5000, 10000, float('inf')]
            labels = ['1천개 미만', '1천개~3천개', '3천개~5천개', '5천개~1만개', '1만개 이상']
            
            # 2024년 1월 ~ 2024년 7월 월평균 판매 수량 계산
            months = ['2024년01월', '2024년02월', '2024년03월', '2024년04월', '2024년05월', '2024년06월', '2024년07월', '2024년08월', '2024년09월']
            available_months = [month for month in months if month in df_sales_range_filtered.columns]
            
            def calculate_mean(row):
                valid_values = [value for value in row if pd.notna(value) and value != 0]
                if valid_values:
                    return sum(valid_values) / len(valid_values)
                return 0
    
        df_sales_range_filtered['월평균 판매수량'] = df_sales_range_filtered[available_months].apply(calculate_mean, axis=1)
        df_sales_range_filtered = df_sales_range_filtered[df_sales_range_filtered['월평균 판매수량'] > 6]
        
        # 판매구간 분리
        df_sales_range_filtered['판매구간'] = pd.cut(
            df_sales_range_filtered['월평균 판매수량'],
            bins=bins,
            labels=labels,
            right=False
            )
    
        # 판매구간별 SKU 수 카운트
        sales_range_counts = df_sales_range_filtered['판매구간'].value_counts().reindex(labels).fillna(0)
        
        # 파이차트 및 테이블 표시
        col41, col42 = st.columns(2)
        
        with col41:
            fig_pie = px.pie(
                names=sales_range_counts.index,
                values=sales_range_counts.values,
                title=f"{selected_subcategory} 판매구간별 SKU 비율 (24년 평균)",
                color_discrete_sequence=['#FF9900', '#666666', '#999999', '#CCCCCC', '#0000CC']
                )
            
            fig_pie.update_traces(
                texttemplate='%{label}: %{value:f}개',
                textposition='outside',
                textfont_size=12,
                textfont_color='black',
                marker_line_color='black',
                marker_line_width=2
                )
            
            fig_pie.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                title={
                    'text': f"{selected_subcategory} 판매구간별 SKU 비율 (24년 평균)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': 'black'}
                    },
                legend=dict(font=dict(size=12, color='black')),
                height=400,
                width=500
                )
            
        st.plotly_chart(fig_pie, use_container_width=True)
          
        sales_range_summary = df_sales_range_filtered.groupby('판매구간').size().reindex(labels, fill_value=0).reset_index(name='SKU 수')
        sales_range_summary['비율 (%)'] = (sales_range_summary['SKU 수'] / sales_range_summary['SKU 수'].sum() * 100).round(1)
            
        # SKU 수 및 비율 포맷팅
        sales_range_summary['SKU 수'] = sales_range_summary['SKU 수'].astype(str) + '개'
        sales_range_summary['비율 (%)'] = sales_range_summary['비율 (%)'].astype(str) + '%'
            
        # HTML로 테이블 작성
        with col42:
            html = sales_range_summary.to_html(
                index=False,
                classes='table table-striped',
                border=0
                )
            
            st.markdown(
                """
                <style>
                .table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: auto;
                    }
                .table th, .table td {
                    padding: 8px;
                    text-align: right;
                    }
                .table th {
                    background-color: lightgrey;
                    color: black;
                    text-align: center;
                    }
                .table td {
                    background-color: white;
                    color: black;
                    }
                .table td:nth-child(1) {
                    text-align: center;
                    }
                .table caption {
                    caption-side: top;
                    font-weight: bold;
                    }
                </style>
                """,
                unsafe_allow_html=True
                )
            
        st.markdown(html, unsafe_allow_html=True)
            
        # 월령별 재고액
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2>월령별 재고액</h2>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
        df_aging_inventory_filtered = df_aging_inventory[df_aging_inventory['카테고리소분류'] == selected_subcategory]

        if not df_aging_inventory_filtered.empty:
           df_aging_inventory_grouped = df_aging_inventory_filtered.groupby('월령')['재고액'].sum().reset_index()
 
           # 재고액을 억 단위로 변환
           df_aging_inventory_grouped['재고액'] = df_aging_inventory_grouped['재고액'] / 100000000
 
           # 재고액 비중 (%) 계산 및 재고액과 함께 표시할 문자열 포맷팅
           df_aging_inventory_grouped['재고액 비중 (%)'] = (df_aging_inventory_grouped['재고액'] / df_aging_inventory_grouped['재고액'].sum() * 100).round(1)
         
        col43, col44 = st.columns(2)
        
        with col43:
            # 막대차트
            fig_bar = px.bar(
                df_aging_inventory_grouped, 
                x='재고액', 
                y='월령', 
                orientation='h',
                title=f"{selected_subcategory} 월령구간별 재고액",
                labels={'재고액': '재고액 (억)', '월령': '월령 구간'},
                text='재고액',
                color_discrete_sequence=['#FF9900', '#666666', '#999999', '#CCCCCC', '#0000CC']
                )
            
            # 텍스트 및 테두리 설정
            fig_bar.update_traces(
                texttemplate='%{text:.2f}억',  # 소수점 두 자리까지 표시하며 억 단위 추가
                textposition='outside',
                textfont=dict(color='black'),  # 레이블 글씨 색깔 검은색
                marker_line_color='black',  # 바 테두리 색깔 검은색
                marker_line_width=1.5  # 테두리 두께 설정
                )
            
            # 최대 재고액을 계산하고, 여기에 여유를 추가하여 X축의 max 값을 설정
            max_value = df_aging_inventory_grouped['재고액'].max() * 1.1  # 최대값에 10% 여유를 추가
            
            # 레이아웃 설정
            fig_bar.update_layout(
                yaxis=dict(categoryorder='category descending'),
                xaxis=dict(range=[0, max_value]),  # X축 범위를 0에서 max_value로 설정
                xaxis_title="재고액 (억)",
                yaxis_title="월령 구간",
                xaxis_title_font_color='black',
                xaxis_tickfont_color='black',
                yaxis_title_font_color='black',
                yaxis_tickfont_color='black',
                paper_bgcolor='white',
                plot_bgcolor='white',
                title={
                    'text': f"{selected_subcategory} 월령구간별 재고액",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': 'black'}
                    },
                height=400,
                width=500
                )
            
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 테이블
        df_aging_inventory_grouped['재고액 비중 (%)'] = (df_aging_inventory_grouped['재고액'] / df_aging_inventory_grouped['재고액'].sum() * 100).round(1)
        df_aging_inventory_grouped['재고액'] = df_aging_inventory_grouped['재고액'].apply(lambda x: f'{x:.1f}억')
        df_aging_inventory_grouped['재고액 비중 (%)'] = df_aging_inventory_grouped['재고액 비중 (%)'].astype(str) + '%'
               
        with col44:
             html_table = df_aging_inventory_grouped.to_html(
                        index=False,
                       classes='table table-striped',
                       border=0
                       )
                   
             st.markdown(
                       """
                       <style>
                       .table {
                           width: 100%;
                           border-collapse: collapse;
                           margin: auto;
                           }
                       .table th, .table td {
                           padding: 8px;
                           text-align: right;
                           }
                       .table th {
                           background-color: lightgrey;
                           color: black;
                           text-align: center;
                           }
                       .table td {
                           background-color: white;
                           color: black;
                           }
                       .table td:nth-child(1) {
                           text-align: center;
                           }
                       .table caption {
                           caption-side: top;
                           font-weight: bold;
                           }
                       </style>
                       """,
                       unsafe_allow_html=True
                           )
        st.markdown(html_table, unsafe_allow_html=True)
        
        
        # 데이터 전처리: '초도발주' 필터링
        df_order_amt = df_order_amt[df_order_amt['초도재발주구분'] == '초도발주']

        # 연도별, 월별 평균 발주 수량 계산
        df_order_amt_grouped = df_order_amt.groupby(['카테고리소분류', '발주년도', '발주년월'])['발주수량'].mean().reset_index()
        df_order_amt_filtered = df_order_amt_grouped[df_order_amt_grouped['카테고리소분류'] == selected_subcategory]

        if not df_order_amt_filtered.empty:
            yearly_avg_order_amt = df_order_amt_filtered.groupby('발주년도')['발주수량'].mean()
            
        # 연도별 초도 발주 수량 출력 
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2>소분류별 초도 발주 현황</h2>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
        
        col51, col52 = st.columns(2)
        
        with col51:
            # 2024년 데이터만 필터링
            yearly_avg_order_amt_2024 = yearly_avg_order_amt[yearly_avg_order_amt.index == '2024년']

            if not yearly_avg_order_amt_2024.empty:
                columns = st.columns(len(yearly_avg_order_amt_2024))

        for idx, (year, avg_qty) in enumerate(yearly_avg_order_amt_2024.items()):
            # Format the number with a comma and no decimal places
            formatted_avg_qty = f"{avg_qty:,.0f}개"
            
            # Place each metric in its corresponding column
            with columns[idx]:
                st.metric(label=f"{year} 평균발주수량", value=formatted_avg_qty)

   
        # 발주수량 구간을 기준으로 품번(SKU)을 분류
        df_order_amt_filtered = df_order_amt[df_order_amt['카테고리소분류'] == selected_subcategory]
        df_order_amt_filtered['발주수량 구간'] = pd.cut(df_order_amt_filtered['발주수량'], bins=bins, labels=labels, right=False)

        # 발주수량 구간별 품번(SKU) 개수 계산
        sku_count_by_order_amt = df_order_amt_filtered.groupby('발주수량 구간')['품번'].nunique().reset_index(name='발주된 SKU 수')
        sku_count_by_order_amt['발주된 SKU 수'] = sku_count_by_order_amt['발주된 SKU 수'].astype(str) + '개'
        total_sku_count = sku_count_by_order_amt['발주된 SKU 수'].str.replace('개', '').astype(int).sum()
        sku_count_by_order_amt['비중'] = sku_count_by_order_amt['발주된 SKU 수'].str.replace('개', '').astype(int) / total_sku_count * 100
        sku_count_by_order_amt['비중'] = sku_count_by_order_amt['비중'].map(lambda x: f"{x:.1f}%")

        html_table2 = sku_count_by_order_amt.to_html(
            index=False,
            classes='table table-striped',
            border=0
        )

        st.markdown(
            """
            <style>
            .table {
                width: 100%;
                border-collapse: collapse;
                margin: auto;
            }
            .table th, .table td {
                padding: 8px;
                text-align: right;
            }
            .table th {
                background-color: lightgrey;
                color: black;
                text-align: center;
            }
            .table td {
                background-color: white;
                color: black;
            }
            .table td:nth-child(1) {
                text-align: center;
            }
            .table caption {
                caption-side: top;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
            
            
        )

        # Resize the title
        st.markdown("<h3 style='font-size: 18px;'> ■ 구간별 초도발주 SKU 수 및 비중</h3>", unsafe_allow_html=True)
        st.markdown(html_table2, unsafe_allow_html=True)
        
        
        # BEST 상품 사진 출력
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2>소분류별 BEST/WORST TOP10</h2>", unsafe_allow_html=True)
        df_best_worst_filtered = df_best_worst[df_best_worst['카테고리소분류'] == selected_subcategory]

        if df_best_worst_filtered.empty:
            st.error(f"선택한 소분류 '{selected_subcategory}'에 대한 데이터가 없습니다.")
        else:
            # 이미지 URL의 기본 경로
            base_image_url = "http://erp3.daiso.co.kr/jsp/f/fd0060s10.jsp?i_gds_num="

            # SKU를 기본 URL에 추가하여 전체 이미지 URL 생성
            df_best_worst_filtered['이미지 URL'] = base_image_url + df_best_worst_filtered['품번'].astype(str)

            # 월별 열 찾기
            month_columns = [col for col in df_best_worst.columns if '년' in col and '월' in col]

            # 가장 최근 월과 그 이전 월 찾기
            if len(month_columns) >= 2:
                most_recent_month = month_columns[-1]
                previous_month = month_columns[-2]
            else:
                st.error("데이터에 월별 열이 부족합니다.")
                previous_month = None

            if previous_month:
                # 최근 월 판매량 기준으로 상위 20개 및 하위 20개 제품 추출
                df_top10 = df_best_worst_filtered.sort_values(by=previous_month, ascending=False).head(10)
                df_worst10 = df_best_worst_filtered[df_best_worst_filtered[previous_month] > 0].sort_values(by=previous_month, ascending=True).head(10)

                # BEST 20 상품 표시
                st.markdown(f"<h3 style = 'font-size: 18px;'> ■ BEST TOP 10  -  {previous_month}</h3>", unsafe_allow_html=True)
                st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
                for index, row in df_top10.iterrows():
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown(
                            f'<img src="{row["이미지 URL"]}" style="width: 200px; height: 150px; object-fit: auto; border: 2px solid black;">',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(f"품번: {row['품번']}")
                        st.markdown(f"품명: {row['품명']}")
                        st.markdown(f'<span style="color: blue;">**월평균 판매량**: {int(row[previous_month]):,}개</span>',unsafe_allow_html=True)
                # WORST 20 상품 표시
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"<h3 style = 'font-size: 18px;'> ■ Worst TOP 10  - {previous_month}</h3>", unsafe_allow_html=True)
                st.markdown("<div style='background-color:#f0f0f0; padding:1px; border-radius:1px;'>", unsafe_allow_html=True)
                for index, row in df_worst10.iterrows():
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown(
                            f'<img src="{row["이미지 URL"]}" style="width: 200px; height: 150px; object-fit: auto; border: 2px solid black;">',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(f"**품번**: {row['품번']}")
                        st.markdown(f"**품명**: {row['품명']}")
                        st.markdown(f'<span style="color: red;">**월평균 판매량**: {int(row[previous_month]):,}개</span>',unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
else:
    st.write("[카테고리 소분류]를 선택하거나 입력시, 전체 현황 확인 가능.")
    
        
