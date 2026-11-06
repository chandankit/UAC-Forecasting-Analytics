"""Descriptive pipeline analytics for the HHS UAC dataset (no Streamlit dependency, so every function is unit-testable).

Modules
    data_loader           validate the raw HHS export and the notebook's cleaned daily table; select reported days
    kpi_calculations      period filters, stock/flow-correct KPIs, daily/weekly/monthly aggregation
    bottleneck_detection  documented threshold rules for backlog and bottleneck alerts (no look-ahead)
    visualizations        Plotly figure builders used by the dashboard's "Pipeline analytics" tab
"""
