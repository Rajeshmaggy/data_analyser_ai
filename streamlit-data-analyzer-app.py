import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
import pandas as pd
import re
import difflib
import csv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Set page config
st.set_page_config(
    page_title="Chart & Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 1rem;
    }
    .highlight-text {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'analyzed_chart' not in st.session_state:
    st.session_state.analyzed_chart = None
if 'all_results' not in st.session_state:
    st.session_state.all_results = []
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'x_axis_value' not in st.session_state:
    st.session_state.x_axis_value = ""
if 'y_axis_value' not in st.session_state:
    st.session_state.y_axis_value = ""
if 'summary_value' not in st.session_state:
    st.session_state.summary_value = ""
if 'mapped_x' not in st.session_state:
    st.session_state.mapped_x = None
if 'mapped_y' not in st.session_state:
    st.session_state.mapped_y = None
if 'column_suggestions_list' not in st.session_state:
    st.session_state.column_suggestions_list = []
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = ""

# App title
st.markdown('<div class="main-header">Chart & Data Analyzer</div>', unsafe_allow_html=True)
st.markdown("Upload a chart image and corresponding data to get AI-powered insights.")

# Function to analyze chart using Gemini
def analyze_chart_with_gemini(image_bytes, api_key):
    st.markdown('<div class="sub-header">Chart Analysis in Progress...</div>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing chart with Gemini..."):
        try:
            genai.configure(api_key=api_key)
            image = Image.open(io.BytesIO(image_bytes))
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = """
            Analyze the line chart in the uploaded image that may contain multiple series/legends. For each series, provide a structured summary using this exact JSON format:
            {
              "x_axis": "Label and range of X axis",
              "y_axis": "Label and range of Y axis",
              "metric_failed": "The key metric name represented on the Y axis",
              "series": [
                {
                  "legend": "Series Name",
                  "spikes_and_dips": [
                    {
                      "event": "Spike/Dip from X=Date1 to X=Date2",
                      "deviation": "Percentage change (e.g., '50%' or '-20%')"
                    }
                  ]
                }
              ]
            }
            Rules:
            - For each consecutive data point in each series, calculate the deviation as:
                deviation % = ((current value - previous value) / previous value) * 100
            - Report spikes as positive deviations and dips as negative deviations.
            - Include the time period (date, hour, etc.) for each event in each series.
            - Only return valid JSON (no markdown or extra commentary).
            """
            response = model.generate_content([prompt, image], stream=True)
            response.resolve()
            return response.text
        except Exception as e:
            st.error(f"Error analyzing chart: {e}")
            return None

# Function to chunk records
def chunk_records(records, chunk_size=100):
    for i in range(0, len(records), chunk_size):
        yield records[i:i + chunk_size]

# Sidebar for API keys
with st.sidebar:
    st.markdown('<div class="sub-header">API Keys</div>', unsafe_allow_html=True)
    gemini_api_key = st.text_input("Enter Gemini API Key", 
                                   value="AIzaSyDixhl96Z9fOQRU1eQuZPc23yupHGIb17w",
                                   type="password")
    groq_api_key = st.text_input("Enter Groq API Key", 
                                 value="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",
                                 type="password")
    
    st.markdown('<div class="sub-header">Model Settings</div>', unsafe_allow_html=True)
    groq_model = st.selectbox(
        "Select Groq Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-instant", "llama-3.1-8b", "llama-3.1-70b"],
        index=0
    )
    
    st.markdown('<div class="sub-header">Analysis Settings</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk Size", min_value=10, max_value=500, value=100, step=10)
    
    if st.button("Reset Analysis"):
        st.session_state.analyzed_chart = None
        st.session_state.all_results = []
        st.session_state.uploaded_data = None
        st.session_state.x_axis_value = ""
        st.session_state.y_axis_value = ""
        st.session_state.summary_value = ""
        st.session_state.mapped_x = None
        st.session_state.mapped_y = None
        st.session_state.column_suggestions_list = []
        st.session_state.final_summary = ""
        st.experimental_rerun()

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Chart Analysis", "Data Analysis", "Insights"])

# Tab 1: Chart Analysis
with tab1:
    st.markdown('<div class="sub-header">Upload Chart Image</div>', unsafe_allow_html=True)
    uploaded_chart = st.file_uploader("Upload a chart image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_chart is not None:
        # Display the uploaded chart
        st.image(uploaded_chart, caption="Uploaded Chart", use_column_width=True)
        
        # Button to analyze chart
        if st.button("Analyze Chart"):
            chart_bytes = uploaded_chart.getvalue()
            result_text = analyze_chart_with_gemini(chart_bytes, gemini_api_key)
            
            if result_text:
                st.session_state.analyzed_chart = result_text
                
                try:
                    json_match = re.search(r'\{[\s\S]*\}', result_text)
                    if json_match:
                        clean_json = json_match.group(0)
                        data = json.loads(clean_json)
                        
                        # Filter spikes/dips with deviation >= 50%
                        for series in data.get("series", []):
                            series["spikes_and_dips"] = [
                                e for e in series.get("spikes_and_dips", [])
                                if abs(float(e["deviation"].replace('%', '').strip())) >= 50
                            ]
                        
                        data["file"] = uploaded_chart.name
                        st.session_state.all_results = [data]
                        
                        # Generate summary
                        summary_lines = []
                        for result in st.session_state.all_results:
                            chart_file = result["file"]
                            x_axis = result["x_axis"]
                            y_axis = result["y_axis"]
                            
                            summary_lines.append(f"\n📈 Chart: {chart_file}")
                            summary_lines.append(f"X-axis: {x_axis}")
                            summary_lines.append(f"Y-axis: {y_axis}")
                            
                            for series in result.get("series", []):
                                legend = series.get("legend", "")
                                events = series.get("spikes_and_dips", [])
                                
                                if events:
                                    summary_lines.append(f"\n  🔹 Series: {legend}")
                                    for event in events:
                                        summary_lines.append(f"    - {event['event']} with a deviation of {event['deviation']}")
                                else:
                                    summary_lines.append(f"\n  🔹 Series: {legend} — No significant spikes or dips (≥ 50%)")
                        
                        summary_insights = "\n".join(summary_lines)
                        
                        # Store in session state
                        if st.session_state.all_results:
                            st.session_state.x_axis_value = st.session_state.all_results[0]["x_axis"]
                            st.session_state.y_axis_value = st.session_state.all_results[0]["y_axis"]
                            st.session_state.summary_value = summary_insights.strip()
                        
                        # Display results
                        st.markdown('<div class="sub-header">Chart Analysis Results</div>', unsafe_allow_html=True)
                        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                        st.write("**X-axis:**", st.session_state.x_axis_value)
                        st.write("**Y-axis:**", st.session_state.y_axis_value)
                        st.write("**Summary:**")
                        st.write(st.session_state.summary_value)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("No valid JSON found in Gemini output")
                except Exception as e:
                    st.error(f"Error parsing result: {e}")
    
    if st.session_state.analyzed_chart:
        st.markdown('<div class="sub-header">Raw Analysis</div>', unsafe_allow_html=True)
        with st.expander("View Raw Gemini Output"):
            st.code(st.session_state.analyzed_chart)

# Tab 2: Data Analysis
with tab2:
    st.markdown('<div class="sub-header">Upload CSV Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.x_axis_value or not st.session_state.y_axis_value:
        st.warning("Please analyze a chart first to identify X and Y axes")
    
    uploaded_data = st.file_uploader("Upload CSV data file", type=["csv"])
    
    if uploaded_data is not None:
        try:
            # Read file with correct delimiter detection
            content = io.StringIO(uploaded_data.getvalue().decode("utf-8"))
            try:
                dialect = csv.Sniffer().sniff(content.read(1024))
                content.seek(0)
                delimiter = dialect.delimiter
            except:
                delimiter = ','
            
            # Reset file pointer and read CSV
            content.seek(0)
            df = pd.read_csv(content, delimiter=delimiter)
            st.session_state.uploaded_data = df
            
            st.success(f"Successfully loaded: {uploaded_data.name}")
            
            # Display DataFrame preview
            st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Match columns
            st.markdown('<div class="sub-header">Column Matching</div>', unsafe_allow_html=True)
            
            # Clean labels for matching
            if st.session_state.x_axis_value and st.session_state.y_axis_value:
                x_axis_label = st.session_state.x_axis_value.split('(')[0].strip()
                y_axis_label = st.session_state.y_axis_value.split('(')[0].strip()
                
                st.write(f"**Chart X-Axis:** {x_axis_label}")
                st.write(f"**Chart Y-Axis:** {y_axis_label}")
                
                # Fuzzy matching function
                def match_column(gemini_label, df_columns):
                    matches = difflib.get_close_matches(
                        gemini_label.lower(), [col.lower() for col in df_columns], n=1, cutoff=0.5
                    )
                    if matches:
                        matched_index = [col.lower() for col in df_columns].index(matches[0])
                        return df_columns[matched_index]
                    return None
                
                # Match Gemini labels to actual CSV column names
                st.session_state.mapped_x = match_column(x_axis_label, df.columns)
                st.session_state.mapped_y = match_column(y_axis_label, df.columns)
                
                st.write(f"**Mapped X-Axis Column in CSV:** {st.session_state.mapped_x if st.session_state.mapped_x else 'Not found'}")
                st.write(f"**Mapped Y-Axis Column in CSV:** {st.session_state.mapped_y if st.session_state.mapped_y else 'Not found'}")
                
                # Manual override options
                st.markdown('<div class="sub-header">Manual Column Mapping (Optional)</div>', unsafe_allow_html=True)
                manual_x = st.selectbox("Override X-Axis Column", ["--Keep Automatic--"] + list(df.columns), index=0)
                manual_y = st.selectbox("Override Y-Axis Column", ["--Keep Automatic--"] + list(df.columns), index=0)
                
                if manual_x != "--Keep Automatic--":
                    st.session_state.mapped_x = manual_x
                if manual_y != "--Keep Automatic--":
                    st.session_state.mapped_y = manual_y
                
                # Columns excluding Y-axis
                if st.session_state.mapped_y:
                    columns_without_y_axis = [col for col in df.columns if col != st.session_state.mapped_y]
                    st.markdown('<div class="sub-header">Columns for Analysis</div>', unsafe_allow_html=True)
                    st.write(f"Found {len(columns_without_y_axis)} columns (excluding Y-axis)")
                    
                    # Get relevant columns with Groq
                    if st.button("Find Relevant Columns"):
                        with st.spinner("Finding relevant columns with Groq..."):
                            preview_data = df.head(3).to_string(index=False)
                            
                            auto_prompt_template = PromptTemplate(
                                input_variables=["preview", "y_axis", "columns_without_y_axis"],
                                template="""
                                You are a data analyst.

                                Here is a sample from a dataset:
                                {preview}

                                The dataset has these columns:
                                {columns_without_y_axis}

                                The target metric to analyze is: {y_axis}

                                From the list of columns, identify which ones are most relevant to explaining or influencing the target metric "{y_axis}".
                                including quantity performance **over time** (e.g., monthly or yearly analysis)

                                List the column names only. Do not explain.
                                """
                            )
                            
                            llm = ChatGroq(
                                groq_api_key=groq_api_key,
                                model_name=groq_model,
                                temperature=0
                            )
                            
                            chain = LLMChain(llm=llm, prompt=auto_prompt_template)
                            
                            column_suggestions = chain.run(
                                preview=preview_data,
                                y_axis=st.session_state.mapped_y,
                                columns_without_y_axis=", ".join(columns_without_y_axis)
                            )
                            
                            # Clean column names from the suggestion string
                            column_suggestions_list = [
                                re.sub(r'^\s*\d+\.\s*', '', line).strip()
                                for line in column_suggestions.splitlines()
                                if line.strip()
                            ]
                            
                            # Store in session state
                            st.session_state.column_suggestions_list = column_suggestions_list
                            
                            st.markdown('<div class="sub-header">Suggested Relevant Columns</div>', unsafe_allow_html=True)
                            st.write(column_suggestions_list)
                    
                    # Show stored column suggestions if available
                    if st.session_state.column_suggestions_list:
                        st.markdown('<div class="sub-header">Selected Columns for Analysis</div>', unsafe_allow_html=True)
                        
                        # Allow user to select/deselect columns
                        selected_columns = st.multiselect(
                            "Edit column selection",
                            options=columns_without_y_axis,
                            default=st.session_state.column_suggestions_list
                        )
                        
                        if selected_columns:
                            st.session_state.column_suggestions_list = selected_columns
                            
                            # Filter DataFrame using selected columns
                            try:
                                valid_columns = [col for col in selected_columns if col in df.columns]
                                if valid_columns:
                                    filtered_df = df[valid_columns]
                                    st.dataframe(filtered_df.head())
                                    
                                    # Show button to run analysis if all prerequisites are met
                                    if st.button("Run Analysis"):
                                        with st.spinner("Running analysis with Groq..."):
                                            # Prepare records for analysis
                                            records = [
                                                ", ".join(f"{col}: {row[col]}" for col in valid_columns)
                                                for _, row in df.iterrows()
                                            ]
                                            
                                            # Process in chunks
                                            explanations = []
                                            
                                            # Analysis prompt
                                            analysis_prompt = PromptTemplate(
                                                input_variables=["x_axis", "y_axis", "summary", "cleaned_columns", "aggregated_text"],
                                                template="""
                                                You are a data analyst.

                                                Here is some context from a chart summary:
                                                X-axis: {x_axis}
                                                Y-axis: {y_axis}

                                                Summary:
                                                {summary}

                                                You also have tabular data below with these columns:
                                                {cleaned_columns}

                                                ### Task:

                                                Using ONLY the provided tabular data below, explain WHY the insights described in the summary occurred.
                                                Base your explanation on exact numbers and patterns from the columns listed in {cleaned_columns}.

                                                If no clear cause is found from the data, explicitly say: "No clear cause found."

                                                ### Here is the data:
                                                {aggregated_text}
                                                """
                                            )
                                            
                                            llm = ChatGroq(
                                                groq_api_key=groq_api_key,
                                                model_name=groq_model
                                            )
                                            
                                            analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
                                            
                                            progress_bar = st.progress(0)
                                            status_text = st.empty()
                                            
                                            # Process chunks and collect explanations
                                            chunked_records = list(chunk_records(records, chunk_size=chunk_size))
                                            total_chunks = len(chunked_records)
                                            
                                            for i, record_chunk in enumerate(chunked_records):
                                                status_text.text(f"Processing chunk {i+1} of {total_chunks} with {len(record_chunk)} rows...")
                                                progress_bar.progress((i) / total_chunks)
                                                
                                                chunk_text = "\n".join(record_chunk)
                                                
                                                explanation = analysis_chain.run(
                                                    x_axis=st.session_state.x_axis_value,
                                                    y_axis=st.session_state.y_axis_value,
                                                    summary=st.session_state.summary_value,
                                                    cleaned_columns=", ".join(valid_columns),
                                                    aggregated_text=chunk_text
                                                )
                                                
                                                explanations.append(f"Chunk {i+1} Explanation:\n{explanation}")
                                                progress_bar.progress((i + 1) / total_chunks)
                                            
                                            status_text.text("Generating final summary...")
                                            
                                            # Combine all chunk-level explanations
                                            combined_explanations = "\n\n".join(explanations)
                                            
                                            # Final summarization prompt
                                            summary_prompt = PromptTemplate(
                                                input_variables=["combined_explanations"],
                                                template="""
                                                You are a senior data analyst.

                                                You have received the following explanations from multiple data chunks:

                                                {combined_explanations}

                                                ---

                                                ### Final Task:

                                                Analyze the combined explanations and produce ONE concise, cohesive root cause summary in the following structure:

                                                ---

                                                ### 📊 1. Reason for Spike or Dip:
                                                - Clearly state **why** the spike or dip in the Y-axis occurred.
                                                - Use numerical trends or patterns to justify.

                                                ---

                                                ### 🧩 2. Key Contributing Columns:
                                                - List the columns that had the **strongest influence** on the change.
                                                - Rank them by **impact level** if possible.

                                                ---

                                                ### 🔍 3. Why These Columns Matter:
                                                - Explain the **logic or business reasoning** behind why each key column influenced the trend.
                                                - Use clear evidence from the data patterns.

                                                ---

                                                ### ✅ Final Summary:
                                                - Provide a short, cohesive summary tying all causes together.
                                                - Avoid repetition.
                                                - Use bullet points and be as **clear and insightful** as possible.

                                                If no clear cause is found, state: **"No strong root cause found."**
                                                """
                                            )
                                            
                                            summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                                            
                                            # Run final summarization
                                            final_summary = summary_chain.run(combined_explanations=combined_explanations)
                                            
                                            # Store in session state
                                            st.session_state.final_summary = final_summary
                                            
                                            status_text.text("Analysis complete!")
                                            progress_bar.progress(1.0)
                                else:
                                    st.warning("No valid columns selected for analysis")
                            except Exception as e:
                                st.error(f"Error processing columns: {e}")
                else:
                    st.warning("Y-axis column not found or mapped. Cannot proceed with analysis.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Tab 3: Insights
with tab3:
    st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.final_summary:
        st.info("Run the analysis in the 'Data Analysis' tab to see insights here")
    else:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.final_summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export options
        st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
        
        if st.button("Export as Markdown"):
            export_text = f"""
# Chart & Data Analysis Report

## Chart Details
- **X-axis:** {st.session_state.x_axis_value}
- **Y-axis:** {st.session_state.y_axis_value}

## Chart Summary
{st.session_state.summary_value}

## Root Cause Analysis
{st.session_state.final_summary}
            """
            
            st.download_button(
                label="Download Markdown",
                data=export_text,
                file_name="chart_analysis_report.md",
                mime="text/markdown"
            )
        
        # Add JSON export option
        if st.button("Export as JSON"):
            export_json = {
                "chart_details": {
                    "x_axis": st.session_state.x_axis_value,
                    "y_axis": st.session_state.y_axis_value
                },
                "chart_summary": st.session_state.summary_value,
                "root_cause_analysis": st.session_state.final_summary,
                "analyzed_columns": st.session_state.column_suggestions_list
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_json, indent=2),
                file_name="chart_analysis_report.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("Chart & Data Analyzer | Powered by Gemini & Groq")
