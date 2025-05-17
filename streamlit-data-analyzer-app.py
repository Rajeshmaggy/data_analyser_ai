import streamlit as st
from PIL import Image
import io
import pandas as pd
import re
import json
import difflib
import csv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import google.generativeai as genai

# ------------------- CONFIGURATION -------------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your Gemini API key
llm = ChatGroq(groq_api_key="YOUR_GROQ_API_KEY", model_name="llama-3.1-8b-instant")

st.set_page_config(layout="wide")

# ------------------- UI -------------------
left, right = st.columns(2)

with left:
    st.header("📤 Upload Chart Image")
    uploaded_image = st.file_uploader("Upload line chart image", type=["png", "jpg", "jpeg"])

    st.header("📤 Upload CSV Dataset")
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])  

# ------------------- GEMINI CHART ANALYSIS -------------------
def analyze_chart_with_gemini(image_bytes):
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
"""
    response = model.generate_content([prompt, image], stream=True)
    response.resolve()
    return response.text

# ------------------- PROCESSING -------------------
with right:
    st.header("📈 Analysis Output")

    if uploaded_image and uploaded_csv:
        image_bytes = uploaded_image.read()
        chart_response = analyze_chart_with_gemini(image_bytes)

        try:
            match = re.search(r'\{[\s\S]*\}', chart_response)
            if match:
                clean_json = match.group(0)
                chart_data = json.loads(clean_json)

                # Extract axes
                x_axis_value = chart_data.get("x_axis", "")
                y_axis_value = chart_data.get("y_axis", "")
                summary_lines = []
                for series in chart_data.get("series", []):
                    series_name = series.get("legend", "")
                    summary_lines.append(f"Series: {series_name}")
                    for event in series.get("spikes_and_dips", []):
                        deviation = float(event["deviation"].replace('%',''))
                        if abs(deviation) >= 50:
                            summary_lines.append(f"- {event['event']} → {event['deviation']}")
                summary_value = "\n".join(summary_lines)

                # ------------------- CSV PREP -------------------
                content = uploaded_csv.read().decode("utf-8")
                dialect = csv.Sniffer().sniff(content.splitlines()[0])
                delimiter = dialect.delimiter
                df = pd.read_csv(io.StringIO(content), delimiter=delimiter)

                # Fuzzy column mapping
                def match_column(label, columns):
                    matches = difflib.get_close_matches(label.lower(), [c.lower() for c in columns], n=1, cutoff=0.5)
                    if matches:
                        return columns[[c.lower() for c in columns].index(matches[0])]
                    return None

                mapped_x = match_column(x_axis_value.split("(")[0].strip(), df.columns)
                mapped_y = match_column(y_axis_value.split("(")[0].strip(), df.columns)
                columns_without_y = [col for col in df.columns if col != mapped_y]
                preview = df.head(3).to_string(index=False)

                # Suggest relevant columns
                prompt_template = PromptTemplate(
                    input_variables=["preview", "y_axis", "columns_without_y_axis"],
                    template="""
Here is a sample from a dataset:
{preview}

the target metric is: {y_axis}
The other columns: {columns_without_y_axis}
List only the relevant column names to explain the metric.
"""
                )
                chain = LLMChain(llm=llm, prompt=prompt_template)
                suggestions = chain.run(
                    preview=preview,
                    y_axis=mapped_y,
                    columns_without_y_axis=", ".join(columns_without_y)
                )
                suggestion_list = [re.sub(r'^\s*\d+\.\s*', '', s).strip() for s in suggestions.splitlines() if s.strip()]
                filtered_df = df[suggestion_list]

                # Chunk and Analyze
                records = [", ".join(f"{col}: {row[col]}" for col in suggestion_list) for _, row in filtered_df.iterrows()]
                def chunk_records(data, size):
                    for i in range(0, len(data), size):
                        yield data[i:i+size]

                explanations = []
                analysis_prompt = PromptTemplate(
                    input_variables=["x_axis", "y_axis", "summary", "cleaned_columns", "aggregated_text"],
                    template="""
Chart:
X: {x_axis}, Y: {y_axis}

Summary:
{summary}

Columns:
{cleaned_columns}

Data:
{aggregated_text}

Explain why the Y-axis trend happened. Use the columns only.
"""
                )
                for chunk in chunk_records(records):
                    chunk_text = "\n".join(chunk)
                    explanation = LLMChain(llm=llm, prompt=analysis_prompt).run(
                        x_axis=x_axis_value,
                        y_axis=y_axis_value,
                        summary=summary_value,
                        cleaned_columns=", ".join(suggestion_list),
                        aggregated_text=chunk_text
                    )
                    explanations.append(explanation)

                # Final summarization
                summary_prompt = PromptTemplate(
                    input_variables=["combined_explanations"],
                    template="""
You are a data analyst.

From the following explanations:

{combined_explanations}

Create one concise root cause summary using bullet points.
"""
                )
                final_summary = LLMChain(llm=llm, prompt=summary_prompt).run(
                    combined_explanations="\n\n".join(explanations)
                )

                st.subheader("📋 Final Summary")
                st.text_area("Explanation", final_summary, height=400)

            else:
                st.error("No valid JSON found from Gemini output")

        except Exception as e:
            st.error(f"Error processing: {e}")
    else:
        st.info("Please upload both a chart image and a CSV file to begin.")

