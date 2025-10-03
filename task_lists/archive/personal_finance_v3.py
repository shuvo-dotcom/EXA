import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --- Configuration ---
# IMPORTANT: Update this path to where your categorized CSV files are stored.
CATEGORIZED_DATA_PATH = r'C:\Users\ENTSOE\OneDrive\Finance\Bank Statements\Categorised Bank Statements'
INCOME_CATEGORIES = ['Salary', 'Client 1 - ENTSOG', 'Credit', 'Income']

# --- Helper Functions ---

@st.cache_data   
def load_all_data(base_path):
    """
    Loads all categorized transaction data from the specified directory.
    """
    all_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('_categorized.csv'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        return pd.DataFrame()

    df_list = [pd.read_csv(file) for file in all_files]
    if not df_list:
        return pd.DataFrame()
    
    #in the datafiles changes any Value Date column to Date
    for i, df in enumerate(df_list):
        if 'Value Date' in df.columns:
            df.rename(columns={'Value Date': 'Date'}, inplace=True)
        elif 'Booking Date' in df.columns:
            df.rename(columns={'Booking Date': 'Date'}, inplace=True)
        df_list[i] = df

    # make the date formatting consistent some it is dd/mm/yyyy and some is dd-mm-yyyy
    for df in df_list:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df.dropna(subset=['Date'], inplace=True)
        
    df = pd.concat(df_list, ignore_index=True)

    # --- Data Cleaning and Preparation ---
    # Find the date column from a list of possible names
    date_col_found = None
    possible_date_cols = ['Date', 'Value Date', 'Booking Date']
    for col in possible_date_cols:
        if col in df.columns:
            date_col_found = col
            break
    
    if date_col_found:
        # Specify dayfirst=True to correctly parse dd/mm/yyyy format
        df[date_col_found] = pd.to_datetime(df[date_col_found], errors='coerce', dayfirst=True)
        df.dropna(subset=[date_col_found], inplace=True)
        df['Year'] = df[date_col_found].dt.year
        df['Month'] = df[date_col_found].dt.month_name()
    else:
        st.error(f"Could not find a date column. Please ensure one of the following columns exists in your data: {possible_date_cols}")
        return pd.DataFrame()

    # Ensure 'Amount' is a numeric type, checking for common names
    amount_col_found = None
    possible_amount_cols = ['Amount', 'Amount (EUR)']
    for col in possible_amount_cols:
        if col in df.columns:
            amount_col_found = col
            break

    if amount_col_found:
        # Rename to a standard 'Amount' column for consistency
        if amount_col_found != 'Amount':
            df.rename(columns={amount_col_found: 'Amount'}, inplace=True)
        
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df.dropna(subset=['Amount'], inplace=True)
    else:
        st.error(f"Could not find an amount column. Please ensure one of the following columns exists: {possible_amount_cols}")
        return pd.DataFrame()
    return df

def create_summary_table(df, year):
    """
    Creates a pivot table summarizing income and expenses for a given year.
    """
    if df.empty:
        return pd.DataFrame()

    df_year = df[df['Year'] == year].copy()
    
    # Separate income and expenses
    df_year['Type'] = df_year['Category'].apply(lambda c: 'Income' if c in INCOME_CATEGORIES else 'Expense')
    
    # Make expense amounts positive for easier aggregation
    df_year['Amount'] = df_year.apply(lambda row: row['Amount'] if row['Type'] == 'Income' else abs(row['Amount']), axis=1)

    # Create pivot table
    summary = pd.pivot_table(
        df_year, 
        values='Amount', 
        index=['Type', 'Category'], 
        columns='Month', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Order months chronologically
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ordered_months = [m for m in month_order if m in summary.columns]
    summary = summary[ordered_months]
    
    # --- Calculate Totals ---
    # Handle months with no income or expenses gracefully
    income_total = summary.loc['Income'].sum() if 'Income' in summary.index.get_level_values(0) else pd.Series([0]*len(summary.columns), index=summary.columns)
    expense_total = summary.loc['Expense'].sum() if 'Expense' in summary.index.get_level_values(0) else pd.Series([0]*len(summary.columns), index=summary.columns)
    balance = income_total - expense_total
    expense_total = summary.loc['Expense'].sum()
    balance = income_total - expense_total
    
    # Add total rows to the summary
    summary.loc[('Income', 'Total Income'), :] = income_total
    summary.loc[('Expense', 'Total Expenses'), :] = expense_total
    summary.loc[('Balance', 'Monthly Balance'), :] = balance
    
    return summary.sort_index()

# --- Streamlit App Layout ---

st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Personal Finance Dashboard")

# --- Load Data ---
data = load_all_data(CATEGORIZED_DATA_PATH)

if data.empty:
    st.warning(f"No categorized data found in '{CATEGORIZED_DATA_PATH}'.")
    st.stop()

# --- Sidebar for Filters ---
st.sidebar.header("Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(data['Year'].unique(), reverse=True))

# --- Main Content ---

# 1. Summary Table
st.header(f"Financial Summary for {selected_year}")
summary_df = create_summary_table(data, selected_year)
if not summary_df.empty:
    st.dataframe(summary_df.style.format("€{:,.2f}").highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ffcccb'))
else:
    st.info(f"No data available for the year {selected_year}.")

# 2. Charts and Visualizations
st.header("Visualizations")
year_data = data[data['Year'] == selected_year]

if not year_data.empty:
    # --- Chart Filters ---
    chart_type = st.radio("Select Chart Type", ('Bar Chart', 'Pie Chart'))
    
    # Filter for expenses only for clearer charts
    expense_data = year_data[~year_data['Category'].isin(INCOME_CATEGORIES)].copy()
    expense_data['Amount'] = expense_data['Amount'].abs()

    if chart_type == 'Bar Chart':
        st.subheader("Monthly Expenses by Category")
        fig = px.bar(
            expense_data, 
            x='Month', 
            y='Amount', 
            color='Category', 
            title='Monthly Expenses',
            category_orders={"Month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Pie Chart':
        st.subheader("Expense Distribution")
        selected_months = st.multiselect("Select Months to Include", sorted(expense_data['Month'].unique()), default=sorted(expense_data['Month'].unique()))
        
        if selected_months:
            monthly_expense_data = expense_data[expense_data['Month'].isin(selected_months)]
            fig = px.pie(
                monthly_expense_data, 
                names='Category', 
                values='Amount', 
                title=f'Expense Breakdown for Selected Months'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one month to display the pie chart.")
else:
    st.info(f"No data to visualize for {selected_year}.")
