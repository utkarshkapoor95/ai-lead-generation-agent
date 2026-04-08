import streamlit as st
import os, json, time
import pandas as pd
from groq import Groq

import requests

st.set_page_config(page_title="AI Lead Generation Agent", page_icon="🎯", layout="wide")

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
.lead-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.score-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}
.tag {
    display: inline-block;
    background: #f0f4ff;
    color: #4f46e5;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── API Clients ───────────────────────────────────────────
def get_groq():
    key = st.secrets.get("GROQ_API_KEY","") or os.environ.get("GROQ_API_KEY","")
    if not key:
        st.error("❌ Add GROQ_API_KEY to Streamlit secrets"); st.stop()
    return Groq(api_key=key)

def get_tavily_key():
    key = st.secrets.get("TAVILY_API_KEY","") or os.environ.get("TAVILY_API_KEY","")
    if not key:
        st.error("❌ Add TAVILY_API_KEY to Streamlit secrets"); st.stop()
    return key

def get_apify_key():
    return st.secrets.get("APIFY_API_KEY","") or os.environ.get("APIFY_API_KEY","")

# ── MODE 1: Local Business Search (Apify → Google Maps) ──
def search_local_businesses(query: str, location: str, limit: int = 20):
    apify_key = get_apify_key()
    if not apify_key:
        st.warning("⚠️ No Apify key — showing demo data. Add APIFY_API_KEY to secrets for real results.")
        return get_demo_local_leads(query, location, limit)

    search_term = f"{query} in {location}"
    url = f"https://api.apify.com/v2/acts/compass~crawler-google-places/run-sync-get-dataset-items"
    payload = {
        "searchStringsArray": [search_term],
        "maxCrawledPlacesPerSearch": limit,
        "language": "en",
        "maxImages": 0,
        "maxReviews": 0,
        "exportPlaceUrls": False,
    }
    try:
        resp = requests.post(
            url,
            params={"token": apify_key, "timeout": 120},
            json=payload,
            timeout=150
        )
        if resp.status_code == 200:
            data = resp.json()
            leads = []
            for item in data:
                leads.append({
                    "name": item.get("title",""),
                    "category": item.get("categoryName",""),
                    "address": item.get("address",""),
                    "phone": item.get("phone",""),
                    "website": item.get("website",""),
                    "rating": item.get("totalScore",""),
                    "reviews": item.get("reviewsCount",""),
                    "email": "",
                    "type": "local"
                })
            return leads
        else:
            st.warning("Apify error — showing demo data")
            return get_demo_local_leads(query, location, limit)
    except Exception as e:
        st.warning(f"Apify connection issue — showing demo data: {e}")
        return get_demo_local_leads(query, location, limit)

def get_demo_local_leads(query, location, limit):
    demos = []
    for i in range(min(limit, 10)):
        demos.append({
            "name": f"{query.title()} #{i+1} — {location}",
            "category": query.title(),
            "address": f"Sector {i+1}, {location}",
            "phone": f"+91-98{i}00{i}0{i+1}00",
            "website": f"www.{query.lower().replace(' ','')}_{i+1}.com",
            "rating": round(3.5 + (i % 15)/10, 1),
            "reviews": (i+1) * 12,
            "email": f"info@{query.lower().replace(' ','')}_{i+1}.com",
            "type": "local"
        })
    return demos

# ── MODE 2: Startup Research (Tavily) ────────────────────
def search_startups(natural_query: str, limit: int = 20):
    tavily_key = get_tavily_key()
    search_queries = build_startup_queries(natural_query)
    all_results = []

    progress = st.progress(0)
    status = st.empty()

    for i, q in enumerate(search_queries[:3]):
        status.text(f"🔍 Searching: {q}")
        try:
            resp = requests.get(
                "https://api.tavily.com/search",
                params={
                    "api_key": tavily_key,
                    "query": q,
                    "search_depth": "advanced",
                    "max_results": 10,
                    "include_answer": True,
                },
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                all_results.extend(data.get("results", []))
        except Exception as e:
            st.warning(f"Search error: {e}")
        progress.progress((i+1)/3)
        time.sleep(0.5)

    status.empty()
    progress.empty()

    if not all_results:
        return get_demo_startup_leads(natural_query, limit)

    return extract_startup_leads(all_results, natural_query, limit)

def build_startup_queries(query):
    query_lower = query.lower()
    base = query
    queries = [
        f"{base} site:inc42.com OR site:yourstory.com OR site:techcrunch.com",
        f"{base} funding raised 2024 2025 India startup",
        f"{base} recently funded startup crunchbase tracxn",
    ]
    return queries

def extract_startup_leads(results, original_query, limit):
    groq = get_groq()
    content = "\n\n".join([
        f"Source: {r.get('url','')}\nTitle: {r.get('title','')}\nContent: {r.get('content','')[:500]}"
        for r in results[:15]
    ])

    resp = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"""
Extract startup/company leads from this research data.
Original query: {original_query}

Research data:
{content}

Extract up to {limit} companies. Return ONLY valid JSON array:
[{{
  "name": "Company Name",
  "sector": "Industry/Sector",
  "funding": "Amount raised if mentioned",
  "stage": "Seed/Series A/Series B etc",
  "location": "City, Country",
  "website": "website if found",
  "founded": "year if found",
  "description": "1 sentence about what they do",
  "source_url": "url where found",
  "type": "startup"
}}]

Only include companies clearly matching the query. Return [] if none found.
Return ONLY the JSON array, nothing else."""}],
        temperature=0.2,
        max_tokens=2000
    )
    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json","").strip()
    try:
        leads = json.loads(raw)
        return leads if isinstance(leads, list) else []
    except:
        return get_demo_startup_leads(original_query, limit)

def get_demo_startup_leads(query, limit):
    sectors = ["D2C","SaaS","Fintech","Healthtech","Edtech","Agritech","Logistics"]
    stages = ["Seed","Pre-Series A","Series A","Series B"]
    cities = ["Bangalore","Mumbai","Delhi","Hyderabad","Pune","Chennai"]
    demos = []
    for i in range(min(limit, 10)):
        demos.append({
            "name": f"StartupX {i+1}",
            "sector": sectors[i % len(sectors)],
            "funding": f"${(i+1)*2}M",
            "stage": stages[i % len(stages)],
            "location": f"{cities[i % len(cities)]}, India",
            "website": f"www.startupx{i+1}.com",
            "founded": f"202{i%5}",
            "description": f"AI-powered {sectors[i%len(sectors)]} startup solving {query}",
            "source_url": "https://inc42.com",
            "type": "startup"
        })
    return demos

# ── AI: Score + Outreach ─────────────────────────────────
def ai_enrich_leads(leads, context, mode):
    groq = get_groq()
    leads_text = json.dumps(leads[:10], indent=2)

    resp = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"""
You are a lead qualification expert. Enrich these {mode} leads.
User context: {context}

Leads:
{leads_text}

For each lead, add these fields and return enriched JSON array:
- "fit_score": number 1-10 (how good a lead this is)
- "fit_reason": "1 sentence why this score"
- "outreach_email": "personalized cold email subject + 3-line body"
- "outreach_linkedin": "personalized LinkedIn message under 300 chars"
- "pain_points": ["pain1","pain2"]

Return ONLY the enriched JSON array. Keep all original fields, just add the new ones."""}],
        temperature=0.4,
        max_tokens=3000
    )
    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json","").strip()
    try:
        return json.loads(raw)
    except:
        for lead in leads[:10]:
            lead["fit_score"] = 7
            lead["fit_reason"] = "Good potential match based on profile"
            lead["outreach_email"] = "Subject: Quick question\n\nHi, I came across your company and would love to connect. Would you be open to a 15-min call?"
            lead["outreach_linkedin"] = "Hi! I came across your company and think there's a great synergy. Would love to connect!"
            lead["pain_points"] = ["Growth", "Efficiency"]
        return leads

# ── Display Leads ─────────────────────────────────────────
def display_local_lead(lead, i):
    score = lead.get("fit_score", 7)
    color = "#16a34a" if score>=8 else "#d97706" if score>=6 else "#dc2626"
    rating = lead.get("rating","")
    rating_str = f"⭐ {rating}" if rating else ""

    with st.expander(f"#{i+1} {lead.get('name','Unknown')} — Score: {score}/10", expanded=i<3):
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"**📍 Address:** {lead.get('address','N/A')}")
            if lead.get('phone'):
                st.markdown(f"**📞 Phone:** `{lead.get('phone')}`")
            if lead.get('website'):
                st.markdown(f"**🌐 Website:** {lead.get('website')}")
            if lead.get('email'):
                st.markdown(f"**📧 Email:** {lead.get('email')}")
            st.markdown(f"**🏷️ Category:** {lead.get('category','N/A')} {rating_str}")
        with col2:
            st.markdown(f"""
            <div style='background:{color}15;border:1px solid {color}40;
            border-radius:10px;padding:12px;text-align:center'>
            <div style='font-size:2rem;font-weight:900;color:{color}'>{score}</div>
            <div style='font-size:11px;color:#6b7280'>Lead Score</div>
            </div>""", unsafe_allow_html=True)
            st.caption(lead.get("fit_reason",""))

        if lead.get("outreach_email"):
            st.markdown("**✉️ Outreach Email:**")
            st.code(lead.get("outreach_email"), language=None)
        if lead.get("outreach_linkedin"):
            st.markdown("**💼 LinkedIn Message:**")
            st.code(lead.get("outreach_linkedin"), language=None)

def display_startup_lead(lead, i):
    score = lead.get("fit_score", 7)
    color = "#16a34a" if score>=8 else "#d97706" if score>=6 else "#dc2626"

    with st.expander(f"#{i+1} {lead.get('name','Unknown')} — {lead.get('stage','')} — Score: {score}/10", expanded=i<3):
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"**🏢 Sector:** {lead.get('sector','N/A')}")
            st.markdown(f"**💰 Funding:** {lead.get('funding','N/A')}")
            st.markdown(f"**📍 Location:** {lead.get('location','N/A')}")
            if lead.get('website'):
                st.markdown(f"**🌐 Website:** {lead.get('website')}")
            if lead.get('founded'):
                st.markdown(f"**📅 Founded:** {lead.get('founded')}")
            st.markdown(f"**📝 About:** {lead.get('description','N/A')}")
            if lead.get('source_url'):
                st.markdown(f"**🔗 Source:** [{lead.get('source_url')}]({lead.get('source_url')})")
        with col2:
            st.markdown(f"""
            <div style='background:{color}15;border:1px solid {color}40;
            border-radius:10px;padding:12px;text-align:center'>
            <div style='font-size:2rem;font-weight:900;color:{color}'>{score}</div>
            <div style='font-size:11px;color:#6b7280'>Lead Score</div>
            </div>""", unsafe_allow_html=True)
            st.caption(lead.get("fit_reason",""))

        if lead.get("outreach_email"):
            st.markdown("**✉️ Outreach Email:**")
            st.code(lead.get("outreach_email"), language=None)
        if lead.get("outreach_linkedin"):
            st.markdown("**💼 LinkedIn Message:**")
            st.code(lead.get("outreach_linkedin"), language=None)

# ── Export CSV ────────────────────────────────────────────
def leads_to_csv(leads):
    df = pd.DataFrame(leads)
    return df.to_csv(index=False).encode("utf-8")

# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#4f46e5,#7c3aed);
color:white;border-radius:14px;padding:24px 28px;margin-bottom:24px'>
<h1 style='margin:0;font-size:2rem'>🎯 AI Lead Generation Agent</h1>
<p style='margin:6px 0 0;opacity:.85'>
Find local businesses OR funded startups → AI scores & writes outreach → Export CSV
</p>
</div>
""", unsafe_allow_html=True)

# Mode selector
mode = st.radio(
    "**Choose Mode:**",
    ["🏪 Local Business Leads", "🚀 Startup & Company Leads"],
    horizontal=True
)

st.divider()

if mode == "🏪 Local Business Leads":
    st.markdown("### 🏪 Find Local Business Leads")
    st.caption("Find any type of business in any city — get phone, address, website, rating")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        business_type = st.text_input(
            "Business Type",
            placeholder="e.g. dentists, CA firms, restaurants, gyms",
        )
    with col2:
        location = st.text_input(
            "Location",
            placeholder="e.g. Delhi, Mumbai, Bangalore"
        )
    with col3:
        limit = st.number_input("Max Results", min_value=5, max_value=50, value=20)

    context = st.text_input(
        "Your context (optional)",
        placeholder="e.g. I am selling dental software, targeting clinics with 2+ dentists"
    )

    if st.button("🚀 Find Leads", type="primary",
                 disabled=not (business_type and location)):
        with st.spinner(f"Searching for {business_type} in {location}..."):
            leads = search_local_businesses(business_type, location, limit)

        if leads:
            st.success(f"✅ Found {len(leads)} businesses. AI is scoring and writing outreach...")
            with st.spinner("AI enriching leads..."):
                leads = ai_enrich_leads(leads, context or f"Finding {business_type} leads in {location}", "local business")

            leads_sorted = sorted(leads, key=lambda x: x.get("fit_score",0), reverse=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Leads", len(leads_sorted))
            high = sum(1 for l in leads_sorted if l.get("fit_score",0)>=8)
            col2.metric("High Quality (8+)", high)
            col3.metric("Avg Score", round(sum(l.get("fit_score",0) for l in leads_sorted)/len(leads_sorted),1))

            st.download_button(
                "📥 Download CSV",
                leads_to_csv(leads_sorted),
                file_name=f"{business_type}_{location}_leads.csv",
                mime="text/csv",
                type="primary"
            )
            st.divider()

            for i, lead in enumerate(leads_sorted):
                display_local_lead(lead, i)
        else:
            st.error("No results found. Try a different search.")

else:
    st.markdown("### 🚀 Find Startup & Company Leads")
    st.caption("Use natural language — AI searches Inc42, YourStory, TechCrunch, Crunchbase")

    examples = [
        "Indian D2C startups that raised funding in last 6 months",
        "SaaS startups in Bangalore with Series A funding 2024",
        "Fintech startups in India funded by Sequoia or Accel",
        "Healthtech startups in India with less than 200 employees",
        "AI startups in India that raised seed funding in 2025",
    ]

    st.markdown("**Examples:**")
    cols = st.columns(len(examples))
    selected_example = None
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex[:30]+"...", key=f"ex{i}", help=ex):
                selected_example = ex

    query = st.text_area(
        "Your Lead Query",
        value=selected_example or "",
        placeholder='e.g. "Find me 30 Indian D2C startups that raised funding in last 6 months with less than 200 employees"',
        height=80
    )

    col1, col2 = st.columns([3,1])
    with col2:
        limit2 = st.number_input("Max Results", min_value=5, max_value=50, value=20, key="limit2")

    context2 = st.text_input(
        "Your context (optional)",
        placeholder="e.g. I am a B2B SaaS selling HR tools to funded startups",
        key="ctx2"
    )

    if st.button("🚀 Find Startups", type="primary", disabled=not query.strip()):
        with st.spinner("AI agent searching across Inc42, YourStory, TechCrunch..."):
            leads = search_startups(query, limit2)

        if leads:
            st.success(f"✅ Found {len(leads)} companies. AI scoring and writing outreach...")
            with st.spinner("AI enriching leads..."):
                leads = ai_enrich_leads(leads, context2 or query, "startup")

            leads_sorted = sorted(leads, key=lambda x: x.get("fit_score",0), reverse=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Leads", len(leads_sorted))
            high = sum(1 for l in leads_sorted if l.get("fit_score",0)>=8)
            col2.metric("High Quality (8+)", high)
            col3.metric("Avg Score", round(sum(l.get("fit_score",0) for l in leads_sorted)/len(leads_sorted),1) if leads_sorted else 0)

            st.download_button(
                "📥 Download CSV",
                leads_to_csv(leads_sorted),
                file_name="startup_leads.csv",
                mime="text/csv",
                type="primary"
            )
            st.divider()

            for i, lead in enumerate(leads_sorted):
                display_startup_lead(lead, i)
        else:
            st.warning("No startups found. Try a broader query.")

st.divider()
st.caption("🎯 AI Lead Generation Agent | Powered by Groq + Tavily + Apify | Built for sales, outreach & market research")
