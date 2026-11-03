import json,pandas as pd,streamlit as st,plotly.express as px
from config import RAW,ART
from src.database import init,save,history as get_history
from src.inference import Predictor
st.set_page_config(page_title="UPI IntentGuard",page_icon="🛡️",layout="wide")
st.markdown('''<style>.stApp{background:linear-gradient(130deg,#071426,#102d4a);color:#eef6ff} [data-testid="stSidebar"]{background:#08172a} h1,h2,h3,p,label{color:#eef6ff} div[data-testid="stMetric"]{background:#123252;padding:14px;border-radius:12px;border:1px solid #285174}</style>''',unsafe_allow_html=True)
init()
st.sidebar.title("🛡️ UPI IntentGuard")
page=st.sidebar.radio("Workspace",["Overview","Live Risk Assessment","Transaction Explorer","Model Evaluation","Assessment History"])
st.sidebar.caption("Synthetic research prototype")
@st.cache_data
def data():return pd.read_csv(RAW) if RAW.exists() else pd.DataFrame()
@st.cache_resource
def predictor():return Predictor()
df=data()
st.title("UPI IntentGuard")
st.caption("Intent-aware detection of potentially manipulated, authorized digital payments")
if page=="Overview":
    st.markdown("### Security intelligence overview")
    if df.empty:st.warning("Generate data first: `python -m src.data.generate_data`")
    else:
        a,b,c,d=st.columns(4); a.metric("Transactions",f"{len(df):,}"); b.metric("Simulated cases",f"{int(df.is_fraud.sum()):,}")
        c.metric("Users",f"{df.user_id.nunique():,}"); d.metric("Recipients",f"{df.recipient_id.nunique():,}")
        x,y=st.columns(2)
        with x:st.plotly_chart(px.pie(df.is_fraud.value_counts().rename({0:"Genuine",1:"Simulated manipulation"}).rename_axis("Class").reset_index(name="Count"),names="Class",values="Count",title="Synthetic label distribution",hole=.5),use_container_width=True)
        with y:st.plotly_chart(px.histogram(df,x="amount",color="is_fraud",title="Payment amount distribution"),use_container_width=True)
        st.subheader("Recent transactions"); st.dataframe(df.tail(15),use_container_width=True,hide_index=True)
    st.info("The system compares a payment with behavioural sequences, recipient relationships and payment context.")
elif page=="Live Risk Assessment":
    st.subheader("Assess a fictional payment")
    st.warning("Do not enter real account numbers, UPI PINs, passwords or personal information.")
    x,y=st.columns(2)
    with x:
        default_user = str(df.user_id.iloc[0]) if not df.empty else "DEMO_USER_01"
        user=st.text_input("Demo user reference", default_user); amount=st.number_input("Amount (₹)",1.,1000000.,8500.,100.)
        recipient=st.text_input("Recipient reference","NEW_RECIPIENT_77")
        category=st.selectbox("Payment category",["transfer","food","groceries","utilities","shopping","travel","investment","urgent_transfer"])
    with y:
        hour=st.slider("Payment hour",0,23,1); device=st.selectbox("Device",["D0","D1","D2","D4","D5"]); location=st.selectbox("Location",["L0","L1","L3","L16","L20"])
    if st.button("Analyse payment",type="primary",use_container_width=True):
        try:
            required = [
    "amount",
    "timestamp",
    "recipient_id",
    "category",
    "device_type",
    "location"
]

            if not df.empty:
                # Retrieve the user's COMPLETE transaction history
                hist = df.loc[
                    df["user_id"].astype(str) == str(user),
                    required
                ].copy()

                # Sort transactions chronologically
                hist["timestamp"] = pd.to_datetime(hist["timestamp"])

                hist = hist.sort_values("timestamp").reset_index(drop=True)

            else:
                hist = pd.DataFrame(columns=required)
            
            if hist.empty:
                st.info("No past transactions found for this demo user. The assessment will use a new-user baseline. Try a user ID shown in the Transaction Explorer to test historical behaviour.")
            result=predictor().predict(amount,hour,recipient,category,device,location,hist)
            save(user,recipient,amount,result["risk"],result["ids"],result["action"],result["probability"])

            st.markdown("### Risk indicators")
            indicator_cols = st.columns(4)
            for col, (label, value) in zip(indicator_cols, result["parts"].items()):
                col.metric(label, f"{value:.1f}%")

            st.markdown("### Model assessment")
            a,b,c=st.columns(3)
            a.metric("Fraud risk",f'{result["risk"]}%')
            b.metric("Intent Drift Score",f'{result["ids"]}/100')
            c.metric("Model probability",f'{result["probability"]}%')
            if result["action"]=="PROTECT":
                st.error("🛑 HIGH-RISK PAYMENT — PROTECT. " + result["message"])
            elif result["action"]=="VERIFY":
                st.warning("⚠️ SUSPICIOUS PAYMENT — VERIFY. " + result["message"])
            else:
                st.success("✅ LOW-RISK PAYMENT — ALLOW. " + result["message"])
            parts=pd.DataFrame({"Factor":result["parts"].keys(),"Score":result["parts"].values()})
            st.plotly_chart(px.bar(parts,x="Score",y="Factor",orientation="h",range_x=[0,100],title="Behaviour, recipient and sequence indicators"),use_container_width=True)
        except Exception as e:st.exception(e)
elif page=="Transaction Explorer":
    st.subheader("Explore generated transaction records")
    if df.empty:st.warning("Generate the dataset first.")
    else:
        cls=st.multiselect("Class",["Genuine","Simulated manipulation"],["Genuine","Simulated manipulation"])
        d=df.copy();d["Class"]=d.is_fraud.map({0:"Genuine",1:"Simulated manipulation"});d=d[d.Class.isin(cls)]
        st.metric("Visible records",f"{len(d):,}");st.dataframe(d,use_container_width=True,hide_index=True)
        st.download_button("Download CSV",d.to_csv(index=False),"intentguard_synthetic_data.csv","text/csv")
elif page=="Model Evaluation":
    p=ART/"metrics.json"
    if not p.exists():st.warning("Train the model first: `python -m src.training.train`")
    else:
        m=json.loads(p.read_text());st.warning(m["note"])
        cols=st.columns(5)
        for col,k in zip(cols,["accuracy","precision","recall","f1","roc_auc"]):col.metric(k.replace("_"," ").title(),f'{m[k]:.3f}')
        cm=pd.read_csv("reports/confusion_matrix.csv",index_col=0,header=None) if False else pd.read_csv("reports/confusion_matrix.csv")
        st.plotly_chart(px.imshow(cm,text_auto=True,title="Confusion matrix"),use_container_width=True)
        st.code(open("reports/classification_report.txt").read())
else:
    st.subheader("Saved assessments"); h=get_history()
    st.dataframe(h,use_container_width=True,hide_index=True)
    if len(h):st.download_button("Export assessment history",h.to_csv(index=False),"assessment_history.csv","text/csv")
