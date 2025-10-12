import streamlit as st
import pandas as pd
import itertools
import io

# Page configuration (to improve width)
st.set_page_config(layout="wide")

# Application Title
st.title("📊 Frequency Analysis and Cross Tabulations of Job Data")
st.markdown("Upload your CSV or PKL file. This app allows you to select any column with labels (in tuple format) to calculate frequency and perform dynamic cross-tabulations.")
st.divider()

# Initialize empleos_df_final to None
if 'empleos_df_final' not in st.session_state:
    st.session_state.empleos_df_final = None

# --- DEFINITION OF REQUESTED ANALYSIS GROUPS ---

GROUP_A_COLS = ['region', 'sector', 'minimum_qualification_final']
GROUP_B_COLS = ['esco_skill', 'esco_broader_skill', 'esco_skill_broader_hierarchy', 
                'at_least_one_skill_type_skill', 'at_least_one_skill_type_knowledge', 
                'at_least_one_skill_reuse_cross_sector', 'at_least_one_skill_reuse_sect_spec', 
                'at_least_one_skill_reuse_occup_spec', 'at_least_one_skill_reuse_trans']
GROUP_C_COLS = ['isco_final_label', 'isco_final_1digit_label', 'has_isco_0', 'has_isco_1', 
                'has_isco_2', 'has_isco_3', 'has_isco_4', 'has_isco_5', 'has_isco_6', 
                'has_isco_7', 'has_isco_8', 'has_isco_9', 'has_Low_skill', 'has_Medium_skill', 
                'has_High_skill']
GROUP_D_COLS = ['at_least_one_digital_skill', 'at_least_one_green_skill'] 

ALL_DEFINED_COLS = list(set(GROUP_A_COLS + GROUP_B_COLS + GROUP_C_COLS + GROUP_D_COLS))


# List of columns considered 'multi-label' (with tuples)
MULTI_LABEL_COLUMNS = ['region', 'sector', 'esco_skill', 'esco_broader_skill','esco_skill_type',
                       'esco_skill_reuse_level','esco_skill_broader_hierarchy',
                       'type_skill','type_skill2','isco_final','isco_final_1digit','isco_final_skill',
                       'isco_final_label','isco_final_1_digit_label', 'isco_final_1digit_label']
# Ensure new multi-label variables are included in the initial parsing
MULTI_LABEL_COLUMNS = list(set(MULTI_LABEL_COLUMNS + GROUP_B_COLS + GROUP_C_COLS))


# --- List of columns whose percentage is ALWAYS calculated based on the total NON-EMPTY rows ---
# Includes all simple categorical variables where NaN must be excluded from the total (Yes/No, ISCO, Region, Sector, etc.)
COUNT_BY_ROW_COLS = ['region', 'sector', 'at_least_one_skill_type_skill', 
                     'minimum_qualification_final', 'at_least_one_skill_type_knowledge', 
                     'at_least_one_skill_reuse_cross_sector', 'at_least_one_skill_reuse_sect_spec', 
                     'at_least_one_skill_reuse_occup_spec', 'at_least_one_skill_reuse_trans', 
                     'isco_final_label', 'isco_final_1digit_label', 'has_isco_0', 'has_isco_1', 
                     'has_isco_2', 'has_isco_3', 'has_isco_4', 'has_isco_5', 'has_isco_6', 
                     'has_isco_7', 'has_isco_8', 'has_isco_9', 'has_Low_skill', 
                     'has_Medium_skill', 'has_High_skill',
                     'at_least_one_digital_skill', 'at_least_one_green_skill']
# Unify to get the definitive check list
COUNT_BY_ROW_COLUMNS = list(set(COUNT_BY_ROW_COLS)) 

# --- HELPERS FOR COMPLEX DATA PROCESSING ---

def recursive_flatten(iterable):
    """
    Auxiliary function to recursively flatten nested lists, tuples, and sets.
    Ensures only clean strings are returned within a tuple.
    """
    flat_list = []
    # Iterate if it's a known iterable type (list, tuple, set)
    if isinstance(iterable, (list, tuple, set)):
        for element in iterable:
            # Handle recursion and nesting
            if isinstance(element, (list, tuple, set)):
                flat_list.extend(recursive_flatten(element))
            # If the element is not null and not an empty string
            elif pd.notna(element) and str(element).strip():
                flat_list.append(str(element).strip())
    # If it's not an iterable, treat it as a single element (if not null)
    elif pd.notna(iterable) and str(iterable).strip():
        flat_list.append(str(iterable).strip())
        
    # Return a tuple of strings so it's hashable
    return tuple(flat_list)


def parse_item_for_multilabel(item):
    """
    Handles null values, serialized strings, and nested structures (lists of lists)
    to convert the cell content to a flat tuple of strings.
    """
    
    # 1. Safely handle null and empty values (prevents ambiguity error)
    if item is None: 
        return tuple()
    
    # Try to check if the value is null (pd.isnull) or an empty string.
    try:
        if pd.isnull(item) or (isinstance(item, str) and not item.strip()):
            return tuple()
    except ValueError:
        # The ValueError means the element is NOT null and MUST be processed. 
        pass
        
    # 2. If it's a data structure (list, tuple, set) that passed the null check
    if isinstance(item, (list, tuple, set)):
        # Flatten it directly
        return recursive_flatten(item)
        
    # 3. If it's a string, try to evaluate it (if it was serialized)
    if isinstance(item, str):
        item_str = item.strip()
        
        try:
            # Try to evaluate the string (e.g., '("A", "B")' or '["A", "B"]')
            evaluated_item = eval(item_str)
            # If evaluation results in an iterable, flatten it
            if isinstance(evaluated_item, (list, tuple, set)):
                return recursive_flatten(evaluated_item)
            else:
                # If eval returns a scalar (e.g., 'North'), convert it to a single tuple
                return (str(evaluated_item),)
                
        except (NameError, TypeError, SyntaxError):
            # If eval fails (it's just a simple string 'North'), treat it as a single tuple
            return (item_str,) if item_str else tuple()
            
    # 4. Default case (scalar value that is not a string or iterable)
    return (str(item),) # Convert it to string and then to a single tuple

# --- DATA LOADING FUNCTION ---

@st.cache_data(show_spinner="Loading and processing data...")
def load_data(uploaded_file, file_type):
    """Loads data from a CSV or PKL file and processes multi-label columns."""
    df = None
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type == 'pkl':
        # For PKL files, read directly
        df = pd.read_pickle(uploaded_file)
    
    if df is not None:
        # Crucial processing: Use the robust parsing function
        for col in MULTI_LABEL_COLUMNS:
            if col in df.columns:
                # Use .apply() with the new robust function
                df[col] = df[col].apply(parse_item_for_multilabel)
    
    return df

# --- GENERALIZED CALCULATION FUNCTION (COUNT ONLY) ---

@st.cache_data(show_spinner="Calculating frequencies...")
def calculate_frequency(df, column_name):
    """
    Calculates the occurrence frequency count of elements in a column (multi-label).
    Returns the result DataFrame (count only) and the Series of flat items.
    """
    if column_name not in df.columns:
        return pd.DataFrame(), pd.Series()

    # 1. Flatten the column of tuples
    try:
        # Ensure we are only chaining tuples of strings
        all_items = list(itertools.chain.from_iterable(df[column_name]))
    except TypeError:
        st.error(f"Type error while flattening column '{column_name}'. Ensure all cells in that column contain tuples of strings after parsing.")
        return pd.DataFrame(), pd.Series()


    # 2. Convert the list to a Pandas Series and get counts
    items_series = pd.Series(all_items)
    frequency_counts = items_series.value_counts()
    
    # 3. Join results into a sorted DataFrame (COUNT ONLY)
    df_frequency = pd.DataFrame({
        'Count': frequency_counts.astype(int), 
    }).sort_values(by='Count', ascending=False)
    
    # Return the DF (count only) and the Series of all items (for the denominator)
    return df_frequency, items_series 

def format_frequency_df(df_frequency, total_series):
    """Adds a total row to the frequency DataFrame and applies display formatting."""
    
    # Get the total count value (which has already been determined and passed as Series)
    # The total_series contains a single value indexed by 'TOTAL'
    total_conteo_val = total_series.iloc[0]
    
    # Check if the DataFrame has any data to prevent errors if empty
    if df_frequency.empty:
        total_porcentaje = 0.0
    else:
        # Calculate the total percentage (should be 100% or very close)
        total_porcentaje = df_frequency['Percentage (%)'].sum().round(2)
    
    # Create the Total row
    total_row = pd.Series({
        'Count': int(total_conteo_val), 
        'Percentage (%)': total_porcentaje
    }, name='TOTAL')
    
    # Add the Total row to the DataFrame
    df_with_total = pd.concat([df_frequency, pd.DataFrame(total_row).T])
    
    # Apply display formatting to columns
    # The styling here is for a single-level index, using IndexSlice is correct
    df_styled = df_with_total.style.format({
        # Integer format, no decimals, with thousands separator if applicable
        'Count': '{:,.0f}',           
        # Two-decimal format
        'Percentage (%)': '{:.2f}'    
    }).set_properties(
        subset=pd.IndexSlice[['TOTAL'], :], 
        **{'font-weight': 'bold', 'background-color': '#f0f2f6'}
    )
    
    return df_styled, df_with_total # Return the styled DF and the base DF (for the chart)

# --- NEW GROUPED ANALYSIS FUNCTIONS (FOR TOP N PER GROUP) ---

@st.cache_data(show_spinner="Calculating grouped frequencies (Counts)...")
def calculate_grouped_frequency(df, multi_label_col, group_col):
    """
    Calculates grouped frequency of a multi-label column by a grouping column (COUNT ONLY).
    Returns the results indexed by (group_col, multi_label_col).
    """
    # 1. Select and copy relevant columns
    df_temp = df[[multi_label_col, group_col]].copy()
    
    # 2. Filter: Only rows that have a value in the main column (multi_label_col)
    df_temp = df_temp[df_temp[multi_label_col].apply(lambda x: len(x) > 0)]
    
    # 3. Explode the main column (multi_label_col)
    df_temp = df_temp.explode(multi_label_col)
    
    # 4. Explode the grouping column if it is also multi-label (e.g., region)
    if group_col in MULTI_LABEL_COLUMNS:
        df_temp = df_temp.explode(group_col)

    # 5. Clean up: Remove rows where the grouping label is null or empty after explosions
    df_temp = df_temp.dropna(subset=[multi_label_col, group_col])
    
    # 6. Count frequencies: Group by the Group column and then count the main label
    skills_by_group_counts = df_temp.groupby(group_col)[multi_label_col].value_counts()

    # 7. Convert the MultiIndex Series to a DataFrame
    df_results_by_group = skills_by_group_counts.rename('Count').to_frame()
    return df_results_by_group


@st.cache_data(show_spinner="Calculating row denominators...")
def calculate_row_denominators(df, cross_col, main_col):
    """
    Calculates the total number of unique rows/observations (Denominator Mode 2) for each group.
    Only counts rows where 'main_col' is NOT empty AND the row belongs to the 'cross_col' group.
    """
    df_temp = df.copy()
    
    # Filter: Only rows that have a value in the main column (multi_label_col)
    df_temp = df_temp[df_temp[main_col].apply(lambda x: len(x) > 0)]
    
    # Explode the grouping column if necessary
    if cross_col in MULTI_LABEL_COLUMNS:
        df_temp = df_temp.explode(cross_col)
        
    df_temp = df_temp.dropna(subset=[cross_col])

    # Group by the cross_col label and count how many original rows (represented by .size()) contributed
    row_counts = df_temp.groupby(cross_col).size().rename('Total_Rows')
    return row_counts.to_frame()


# --- EXCEL EXPORT HELPERS ---

def to_excel_buffer_single(df_to_export, sheet_name="Analysis_Results"):
    """Converts a single DataFrame to an Excel buffer in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Use float_format to ensure percentages look nice in Excel
        # Rename index to allow proper export
        df_to_export.index.name = 'Group_and_Label' 
        df_to_export.to_excel(writer, sheet_name=sheet_name, float_format='%.2f') 
    processed_data = output.getvalue()
    return processed_data

def to_excel_buffer_multi_sheet(df_counts, df_pct, selected_col, cross_col):
    """Converts two DataFrames (counts and percentages) into an Excel buffer with two sheets, handling name limits."""
    
    # --- FIX: Use simple, short names to comply with the 31-character limit in Excel ---
    count_name = "Counts"
    pct_name = "Percentages"
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Reset index names to allow proper export if they are MultiIndex
        df_counts.index.name = 'Row_Label'
        df_counts.columns.name = 'Col_Label'
        df_pct.index.name = 'Row_Label'
        df_pct.columns.name = 'Col_Label'

        df_counts.to_excel(writer, sheet_name=count_name)
        df_pct.to_excel(writer, sheet_name=pct_name, float_format='%.2f')
    processed_data = output.getvalue()
    return processed_data


# --- SIDEBAR (INPUTS) ---

st.sidebar.header("🛠️ File and Analysis Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload your file (CSV or PKL)", 
    type=["csv", "pkl"]
)

# Lógica de Carga y Procesamiento
if uploaded_file:
    # Determine file type
    file_extension = uploaded_file.name.split('.')[-1]
    
    try:
        # Load data and save to session state
        st.session_state.empleos_df_final = load_data(uploaded_file, file_extension)
        df_loaded = st.session_state.empleos_df_final
        
        # Identify all DataFrame columns
        all_cols = df_loaded.columns.tolist()
        
        st.sidebar.success(f"File loaded ({len(df_loaded)} rows).")

    except Exception as e:
        # Use st.exception() to show the full error for debugging
        st.sidebar.error(f"Error loading or processing the file. Details:")
        st.sidebar.exception(e) 
        st.session_state.empleos_df_final = None # Reset state
        df_loaded = None
else:
    df_loaded = st.session_state.empleos_df_final

# --- NEW: JOB PORTAL FILTER (Must happen after df_loaded is defined) ---

df_analysis = None
selected_portals = None
if df_loaded is not None and 'job_portal' in df_loaded.columns:
    st.sidebar.subheader("Data Subset Filter")
    
    # Get unique portal options (handling potential NaN values robustly)
    portal_options = df_loaded['job_portal'].astype(str).str.strip().unique().tolist()
    portal_options = sorted([p for p in portal_options if p.lower() not in ['nan', ''] and p.strip()])

    selected_portals = st.sidebar.multiselect(
        "Select Job Portals (Subset Data)",
        options=portal_options,
        default=portal_options, # Default to selecting all portals
        help="Filter the data to include only observations from the selected job portals."
    )
    
    if selected_portals:
        # Apply filtering
        # Use .copy() to ensure the subset is distinct and avoids SettingWithCopyWarning
        df_analysis = df_loaded[df_loaded['job_portal'].astype(str).isin(selected_portals)].copy()
        
        st.sidebar.info(f"Filtered data size: {len(df_analysis)} rows.")

        if df_analysis.empty:
            st.warning("The selected filters resulted in zero rows. Analysis cannot proceed.")
            df_analysis = None
    else:
        st.warning("Please select at least one Job Portal to enable data filtering and analysis.")
        df_analysis = None
elif df_loaded is not None:
    # If the column 'job_portal' does not exist, or the file upload fails, use the loaded DF (if available)
    df_analysis = df_loaded


# --- ANALYSIS COLUMN SELECTOR (Uses df_loaded for structure, but prepares to run on df_analysis) ---

selected_col = None
if df_loaded is not None: # Use df_loaded to get the available columns regardless of current filter
    all_cols = df_loaded.columns.tolist()

    # Filter group lists to include only columns present in the DF loaded
    available_group_A_cols = [col for col in GROUP_A_COLS if col in all_cols]
    available_group_B_cols = [col for col in GROUP_B_COLS if col in all_cols]
    available_group_C_cols = [col for col in GROUP_C_COLS if col in all_cols]
    available_group_D_cols = [col for col in GROUP_D_COLS if col in all_cols] 

    # --- 1. Create Group Selectors ---
    
    st.sidebar.subheader("Group 1: Location / Qualification")
    selected_col_A = st.sidebar.selectbox(
        "Structure Variables", 
        options=[''] + available_group_A_cols,
        key='sel_A',
        index=0
    )

    st.sidebar.subheader("Group 2: ESCO Skills")
    selected_col_B = st.sidebar.selectbox(
        "Skill Variables", 
        options=[''] + available_group_B_cols,
        key='sel_B',
        index=0
    )

    st.sidebar.subheader("Group 3: ISCO Classification / Skill Level")
    selected_col_C = st.sidebar.selectbox(
        "Classification Variables", 
        options=[''] + available_group_C_cols,
        key='sel_C',
        index=0
    )

    st.sidebar.subheader("Group 4: Digital and Green Skills") 
    selected_col_D = st.sidebar.selectbox( 
        "Specific Skill Variables", 
        options=[''] + available_group_D_cols,
        key='sel_D',
        index=0
    )

    # --- 2. Determine the Single Analysis Column (Priority: A > B > C > D) ---
    
    if selected_col_A:
        selected_col = selected_col_A
    elif selected_col_B:
        selected_col = selected_col_B
    elif selected_col_C:
        selected_col = selected_col_C
    elif selected_col_D: 
        selected_col = selected_col_D
    
    if selected_col is None or selected_col == '':
        st.sidebar.warning("Select a variable from one of the groups to start the analysis.")
        selected_col = None


    # --- PERCENTAGE CALCULATION SELECTOR (For non-rigid multi-label columns) ---
    # Only show this selector if the selected column is multi-label 
    # AND is NOT a rigid column (e.g., 'region', 'sector', etc. that always use non-null rows).
    if selected_col in MULTI_LABEL_COLUMNS and selected_col not in COUNT_BY_ROW_COLUMNS:
        st.sidebar.subheader("Percentage Calculation")
        percentage_mode = st.sidebar.radio(
            "Percentage Base (%)",
            options=["Total count of label occurrences", "Total count of non-null observations"],
            index=0,
            key='percentage_mode',
            help="Defines whether 100% represents the total count of labels found or the total count of rows containing at least one valid label."
        )
    else:
        # Use a temporary state variable to avoid reference errors
        percentage_mode = None 
        if selected_col in COUNT_BY_ROW_COLUMNS and selected_col is not None:
            st.sidebar.info(f"The percentage for '{selected_col.capitalize()}' always uses the total non-null observations.")


# --- 3. VISUALIZATION OF RESULTS ---

if df_analysis is not None and selected_col is not None:
    
    # === A. SINGLE FREQUENCY ANALYSIS (Selected Column) ===
    
    st.header(f"1. Individual Occurrence Frequency: '{selected_col.capitalize()}'")
    
    # -------------------------------------------------------------
    # COUNTING AND PERCENTAGE CALCULATION LOGIC
    # -------------------------------------------------------------
    
    if selected_col in MULTI_LABEL_COLUMNS:
        
        # Get label counts (MultiIndex) and the flat label series
        df_frequency_result, items_series = calculate_frequency(df_analysis, selected_col)
        
        # --- Denominator Calculation ---
        # 1. Total count of label occurrences (sum of the 'Count' column)
        denominator_total_labels = len(items_series) 
        
        # 2. Total count of non-null observations (rows where there is at least one label)
        non_empty_count = (df_analysis[selected_col].apply(lambda x: len(x) > 0)).sum()
        denominator_non_null_rows = non_empty_count 
        
        
        # --- Denominator Assignment based on column and selector ---
        
        if selected_col in COUNT_BY_ROW_COLUMNS:
            # Rigid columns: Always use non-null rows as denominator.
            denominator = denominator_non_null_rows
            total_for_format = pd.Series([denominator], index=['TOTAL']) 
            st.caption(f"Note: Percentage is calculated based on the total NON-EMPTY rows/observations ({denominator}).")
        
        elif percentage_mode == "Total count of non-null observations":
            # Flexible columns with 'Non-null observations' mode
            denominator = denominator_non_null_rows
            total_for_format = pd.Series([denominator], index=['TOTAL']) 
            st.caption(f"Note: Percentage is calculated based on the total NON-EMPTY rows/observations ({denominator}).")
        
        else: # percentage_mode == "Total count of label occurrences" (DEFAULT for flexible)
            # Flexible columns with 'Label occurrences' mode
            denominator = denominator_total_labels
            total_for_format = pd.Series([denominator], index=['TOTAL'])
            st.caption(f"Note: Percentage is calculated based on the total label occurrences ({denominator}).")
            
        # Calculate final percentage
        df_frequency_result['Percentage (%)'] = (df_frequency_result['Count'] / denominator * 100).round(2)
        
    else:
        # Case 3: Simple label columns (not multi-label)
        
        # --- Specific Logic for simple categorical variables where NaN must be excluded ---
        if selected_col in COUNT_BY_ROW_COLUMNS:
             # Count only NON-null values for the denominator
            df_filtered = df_analysis[pd.notna(df_analysis[selected_col])]
            total_non_null = len(df_filtered)
            
            # Recalculate frequency and percentage WITHOUT NULLS
            frequency_counts = df_filtered[selected_col].astype(str).value_counts()
            frequency_percentage = (frequency_counts / total_non_null) * 100
            
            denominator_for_total = total_non_null
            st.caption(f"Note: Percentage is calculated based on the total NON-NULL rows/observations ({total_non_null}).")
            
        else:
            # Simple categorical variables that MAY include NaN in the total (like 'salary')
            frequency_counts = df_analysis[selected_col].astype(str).value_counts()
            frequency_percentage = df_analysis[selected_col].astype(str).value_counts(normalize=True) * 100
            denominator_for_total = len(df_analysis)
            st.caption(f"Note: Percentage is calculated based on the total rows/observations ({len(df_analysis)}).")
        
        
        df_frequency_result = pd.DataFrame({
            'Count': frequency_counts.astype(int), 
            'Percentage (%)': frequency_percentage.round(2)
        }).sort_values(by='Count', ascending=False)
        
        # The total for the TOTAL row is the denominator used
        total_for_format = pd.Series([denominator_for_total], index=['TOTAL']) 
    
    # -------------------------------------------------------------
    # END OF COUNTING AND PERCENTAGE CALCULATION LOGIC
    # -------------------------------------------------------------
    
    # Format the DataFrame to add the TOTAL row
    df_styled_with_total, df_chart_data = format_frequency_df(df_frequency_result, total_for_format)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Frequency Table")
        # Display the styled DataFrame with the TOTAL row
        st.dataframe(df_styled_with_total, use_container_width=True)
    
    with col2:
        st.subheader("Count Chart")
        # NOTE: Chart is generated using original data (without the TOTAL row)
        st.bar_chart(df_frequency_result['Count'])
        
    st.divider()

    # === B. DYNAMIC CROSS-TABULATION ANALYSIS (WITH TOP N PER GROUP) ---
    
    st.header("2. Grouped Analysis and Cross-Tabulations")
    st.markdown("Generates a Top N analysis of the primary column's labels, grouped by another categorical column. For categorical crosses (e.g., Yes/No vs. Qualification), Frequency and Percentage tables will be shown separately.")
    
    # Available columns for cross-tabulation (all except the analysis column)
    available_cross_cols = [col for col in df_loaded.columns if col != selected_col]
    
    # Use two columns for the selector and the slider
    cross_col_input, top_n_input = st.columns([1, 1])

    # List of groups for the cross-column selector
    group_options = {
        "Group 1: Location / Qualification": [col for col in GROUP_A_COLS if col in available_cross_cols],
        "Group 2: ESCO Skills": [col for col in GROUP_B_COLS if col in available_cross_cols],
        "Group 3: ISCO Classification / Skill Level": [col for col in GROUP_C_COLS if col in available_cross_cols],
        "Group 4: Digital and Green Skills": [col for col in GROUP_D_COLS if col in available_cross_cols]
    }
    
    # Flatten the grouped options for the st.selectbox, creating a list of options with category headers
    grouped_options_flat = ['']
    for group_name, cols in group_options.items():
        if cols:
            grouped_options_flat.append(f"--- {group_name} ---")
            grouped_options_flat.extend(cols)

    cross_col = cross_col_input.selectbox(
        "Select the Grouping Column (Groups)", 
        options=grouped_options_flat,
        index=0,
        help="This column will define the groups (e.g., 'minimum_qualification_final' or 'sector')."
    )

    # Check if the selected option is a column name or a header
    if cross_col and cross_col.startswith('---'):
        cross_col = '' # Ignore group headers

    # Slider for Top N
    top_n = top_n_input.slider(
        "Filter by Top N per Group",
        min_value=1,
        max_value=20, 
        value=5,
        step=1
    )
    
    # New Percentage Mode Selector for Cross-Tabulation
    if cross_col and selected_col in MULTI_LABEL_COLUMNS:
        st.markdown("##### Configuración de Porcentaje")
        cross_tab_percentage_mode = st.radio(
            "Base de Porcentaje (%) para el Análisis Agrupado",
            options=["Total count of label occurrences", "Total count of non-null observations (Rows)"],
            index=0,
            key='cross_tab_percentage_mode',
            help="Define si el 100% representa la suma de todas las etiquetas encontradas en el grupo o el total de filas/observaciones únicas que contribuyeron al grupo."
        )
    else:
        cross_tab_percentage_mode = None
    
    
    if cross_col:
        st.subheader(f"Cross-tabulation of '{selected_col.capitalize()}' by '{cross_col.capitalize()}'")
        
        # Conditional Logic: If the main column is multi-label, use GROUPED analysis (Top N).
        if selected_col in MULTI_LABEL_COLUMNS:
            
            # 1. Calculate the raw counts (MultiIndex: Group, Label)
            df_full_grouped_results = calculate_grouped_frequency(df_analysis, selected_col, cross_col) 
            
            # 2. Calculate Denominators and Percentage
            df_download_results = df_full_grouped_results.copy()
            group_totals_series = None # To store unique totals per group

            if cross_tab_percentage_mode == "Total count of non-null observations (Rows)":
                # Denominator Mode 2: Get unique row counts per group
                df_row_denominators = calculate_row_denominators(df_analysis, cross_col, selected_col)
                group_totals_series = df_row_denominators['Total_Rows']

                # Join denominators to the main results to calculate percentages
                df_results_with_denominators = df_download_results.join(df_row_denominators, on=cross_col)
                
                # Calculate Percentage based on Total_Rows
                df_download_results['Percentage (%)'] = (
                    df_results_with_denominators['Count'] / df_results_with_denominators['Total_Rows'] * 100
                ).round(2)
                caption_mode = "total de filas/observaciones únicas no nulas"
                
            else: # "Total count of label occurrences"
                # Denominator Mode 1: Get sum of labels per group
                group_label_totals = df_download_results.groupby(level=0)['Count'].transform('sum')
                group_totals_series = df_download_results.groupby(level=0)['Count'].sum()
                
                # Calculate Percentage based on Total_Labels
                df_download_results['Percentage (%)'] = (
                    df_download_results['Count'] / group_label_totals * 100
                ).round(2)
                caption_mode = "total de ocurrencias de etiquetas (suma de todas las etiquetas mencionadas) "

            
            # --- 3. ADD A SINGLE TOTAL ROW PER GROUP (WITH CORRECT SORTING) ---
            total_rows_list = []
            # Define a temporary label that sorts after all other labels
            temp_total_label = '__ZZZ_TOTAL' 
            
            if group_totals_series is not None:
                for group_label, total_count in group_totals_series.items():
                    # FIX: Use the temporary label that sorts last alphabetically
                    idx = pd.MultiIndex.from_tuples([(group_label, temp_total_label)], names=[cross_col, selected_col])
                    total_row_data = {
                        'Count': int(total_count), 
                        'Percentage (%)': 100.00
                    }
                    total_rows_list.append(pd.DataFrame(total_row_data, index=idx))
            
            # Concatenate the main results with the new total rows
            if total_rows_list:
                df_with_totals = pd.concat([df_download_results] + total_rows_list)
                # FIX: Sort by both levels (0 and 1) ascendingly. The temporary label '__ZZZ_TOTAL' 
                # will ensure the TOTAL row is always at the end of its group.
                df_download_results = df_with_totals.sort_index(level=[0, 1], ascending=[True, True])

            # --- Ensure column order is correct for download (Count then Percentage) ---
            df_download_results = df_download_results[['Count', 'Percentage (%)']]
            
            # 4. Apply Top N filtering for display
            
            # Identify total rows vs. data rows (level 1 is the skill/label index)
            # FIX: Use the temporary label for identification
            is_total = df_download_results.index.get_level_values(1) == temp_total_label
            df_totals = df_download_results[is_total]
            df_data = df_download_results[~is_total]

            # Apply Top N filtering to data rows 
            # group_keys=False ensures the result is a DataFrame, not a GroupBy object
            df_top_n = df_data.groupby(level=0, group_keys=False).head(top_n)
            
            # Combine Top N data rows with all Total rows for final display
            df_display_results_final = pd.concat([df_top_n, df_totals]).sort_index(level=[0, 1], ascending=[True, True])

            # --- START OF FIX: Convert index to string to prevent Arrow rendering error ---
            # This is the key fix for the Streamlit rendering issue.
            
            # 1. Reset the index to convert MultiIndex levels into columns.
            df_display_fixed = df_display_results_final.reset_index()
            
            # 2. Force string conversion on the columns that were formerly the index.
            #    This ensures there are no mixed data types.
            df_display_fixed[cross_col] = df_display_fixed[cross_col].astype(str)
            df_display_fixed[selected_col] = df_display_fixed[selected_col].astype(str)

            # 3. Set the index back so the table displays correctly.
            df_display_fixed = df_display_fixed.set_index([cross_col, selected_col])
            # --- END OF FIX ---
            
            # FIX: Rename the temporary index label back to 'TOTAL' for display
            df_display_fixed = df_display_fixed.rename(index={temp_total_label: 'TOTAL'}, level=1)


            # Use columns for displaying the Top N table and the Download button
            col_table, col_download = st.columns([3, 1])

            with col_table:
                # --- Display results table (Filtered) ---
                # FIX: Use the corrected DataFrame 'df_display_fixed' instead of the original one.
                st.dataframe(
                    df_display_fixed.style.format({
                        # Integer format
                        'Count': '{:,.0f}', 
                        # Two-decimal format and % symbol
                        'Percentage (%)': '{:.2f}%'
                    }).set_properties(
                        # Apply bold to TOTAL rows - IndexSlice now uses 'TOTAL' after renaming
                        subset=pd.IndexSlice[(slice(None), 'TOTAL'), :], 
                        **{'font-weight': 'bold', 'background-color': '#e0e5ea'}
                    ), 
                    use_container_width=True
                )
                st.caption(f"Tabla Filtrada: Mostrando Top {top_n} resultados por grupo (incluye fila 'TOTAL' por grupo).")
                st.caption(f"El porcentaje se calcula en base al **{caption_mode}** DENTRO de cada grupo ('{cross_col.capitalize()}').")

            with col_download:
                # FIX: Rename the index label back to 'TOTAL' before passing it to the excel buffer function
                df_export_results = df_download_results.rename(index={temp_total_label: 'TOTAL'}, level=1)
                excel_buffer = to_excel_buffer_single(df_export_results, sheet_name="Grouped_Analysis")
                st.download_button(
                    label="Download FULL Results (Excel)",
                    data=excel_buffer,
                    file_name=f"results_cross_tabulation.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='grouped_download'
                )

        else:
            # Case: Simple Column Cross-tabulation (e.g., 'Yes/No' by 'Qualification')
            
            df_crosstab_temp = df_analysis.copy() 
            
            # Determine if the row column (selected_col) is a variable that needs row normalization
            is_row_normalized = selected_col in COUNT_BY_ROW_COLS
            
            # --- NULL HANDLING IN CROSSTAB ---
            if selected_col in COUNT_BY_ROW_COLS:
                df_crosstab_temp = df_crosstab_temp[pd.notna(df_crosstab_temp[selected_col])]
                
            if cross_col in COUNT_BY_ROW_COLS:
                 df_crosstab_temp = df_crosstab_temp[pd.notna(df_crosstab_temp[cross_col])]


            # Logic for the ROW column (selected_col):
            row_index = df_crosstab_temp[selected_col].astype(str)
            
            # Logic for the COLUMN column (cross_col):
            if cross_col in MULTI_LABEL_COLUMNS:
                # Apply explode NOW for the cross column (e.g., region)
                df_crosstab_temp = df_crosstab_temp.explode(cross_col)
                
                # RESET INDEX after EXPLODE
                df_crosstab_temp = df_crosstab_temp.reset_index(drop=True)
                
                # Re-select row_index and col_index after resetting the index
                row_index = df_crosstab_temp[selected_col].astype(str)
                col_index = df_crosstab_temp[cross_col]
            else:
                col_index = df_crosstab_temp[cross_col].astype(str)
                
            
            # --- GENERATE COUNT AND PERCENTAGE TABLES ---
            
            # 1. Frequency Table (Raw Count)
            cross_tab_counts = pd.crosstab(row_index, col_index)
            
            # 2. Percentage Table (Normalized)
            if is_row_normalized:
                # Normalize by COLUMN (index, so each column/group sums to 100%)
                # Use .fillna(0) to avoid NaN in crosstab when a column has zero counts
                cross_tab_pct = (pd.crosstab(row_index, col_index, normalize='columns') * 100).fillna(0)
                caption_text = "Percentage Table: Values sum to 100% PER COLUMN (i.e., within each group of 'Region', 'Qualification', etc.)."
            else:
                # Normalize by the general total (default)
                cross_tab_pct = pd.crosstab(row_index, col_index, normalize='all') * 100
                caption_text = "Percentage Table: Values are percentages of the general total occurrences in the cross-tabulation."
                
            
            # --- DISPLAY THE TWO TABLES SIDE BY SIDE ---
            col_counts, col_pct = st.columns([1, 1])
            col_download_single = st.columns([1, 4])[0] # Use a narrow column for the download button
            
            with col_counts:
                st.subheader("Absolute Frequency (Count)")
                st.dataframe(cross_tab_counts, use_container_width=True)
                
            with col_pct:
                st.subheader("Relative Frequency (%)")
                st.dataframe(cross_tab_pct.style.format("{:.2f}%"), use_container_width=True)
                
            st.caption(caption_text)
            
            # --- DOWNLOAD BUTTON FOR SIMPLE CROSSTAB ---
            with col_download_single:
                excel_buffer = to_excel_buffer_multi_sheet(cross_tab_counts, cross_tab_pct, 
                                                             selected_col=selected_col, 
                                                             cross_col=cross_col)
                st.download_button(
                    label="Download Cross-Tabulation (Excel)",
                    data=excel_buffer,
                    file_name="results_cross_tabulation.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='crosstab_download'
                )


# --- DATA PREVIEW (Expander) ---

if df_analysis is not None:
    with st.expander("View Loaded DataFrame (First Rows)"):
        # CORRECTION: Use use_container_width=True so it takes up the full width
        st.dataframe(df_analysis.head(10), use_container_width=True)
        st.caption("Verify that multi-label columns now contain flat tuples of strings.")
        

# --- INITIAL INSTRUCTIONS ---

if df_loaded is None:
    st.warning("Please upload your file (CSV or PKL) using the uploader in the left sidebar to start the analysis.")

