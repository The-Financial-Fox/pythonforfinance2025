# Manufacturing CFO AI Dashboard

## Local setup
1. Create a GitHub repository.
2. Add these files to the repo root:
   - `streamlit_cfo_dashboard_app.py`
   - `requirements.txt`
   - `manufacturing_company_dataset(3).xlsx` or place it in `/data` and adjust `DEFAULT_FILE` if desired.
3. Run locally:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_cfo_dashboard_app.py
```

## Publish to Streamlit Community Cloud
1. Push the repo to GitHub.
2. Go to Streamlit Community Cloud and choose **New app**.
3. Select your GitHub repo, branch, and main file path: `streamlit_cfo_dashboard_app.py`.
4. Click **Deploy**.
5. To update the dashboard, push changes to GitHub; Streamlit redeploys automatically.

## Notes
- The dashboard can load the included Excel file automatically or accept a replacement Excel/CSV upload from the sidebar.
- Working capital is proxied by inventory value because AR/AP are not included in the dataset.
- AI insights are deterministic FP&A commentary generated from business rules, not an external LLM API.
