import sys, os, json, pickle, numpy as np, pandas as pd
os.makedirs("figures", exist_ok=True)
sys.path.insert(0, ".")
import hub_exposure as H
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b","xtick.color":"#52514e","ytick.color":"#52514e"})
cfg=json.load(open("config_medhub.json"))
try: G=pickle.load(open("G.pkl","rb"))
except Exception:
    G=H.build_network(cfg); pickle.dump(G,open("G.pkl","wb"))
E=pd.read_csv("results/table_link_criticality.csv"); D=pd.read_csv("results/table_destination_exposure.csv")
S=pd.read_csv("results/table_scenarios.csv")
BLUE="#2a78d6"; ORANGE="#eb6834"; GREY="#c9c8c3"; INK="#0b0b0b"
def base(ax, lw_road=0.35, lw_rail=0.6, extent=None):
    for u,v,d in G.edges(data=True):
        if d["mode"]=="transfer": continue
        x=[G.nodes[u]["x"],G.nodes[v]["x"]]; y=[G.nodes[u]["y"],G.nodes[v]["y"]]
        if extent and not (extent[0]<=x[0]<=extent[1] and extent[2]<=y[0]<=extent[3]): continue
        ax.plot(x,y,color=GREY if d["mode"]=="road" else "#9ec5f4",lw=lw_road if d["mode"]=="road" else lw_rail,zorder=1,solid_capstyle="round")
def points(ax, labels=True, extent=None, offs=None):
    ax.scatter([cfg["hub"]["lon"]],[cfg["hub"]["lat"]],marker="*",s=220,color=INK,zorder=6,edgecolor="white",lw=0.8)
    if labels: ax.annotate("MedHub /\nTanger Med",(cfg["hub"]["lon"],cfg["hub"]["lat"]),xytext=(6,4),textcoords="offset points",fontsize=8,fontweight="bold",ha="left")
    seen=set()
    for d in cfg["destinations"]:
        k=(d["lat"],d["lon"])
        nm=d["name"].split(" (")[0]
        if nm in seen: continue
        seen.add(nm)
        if extent and not (extent[0]<=d["lon"]<=extent[1] and extent[2]<=d["lat"]<=extent[3]): continue
        ax.scatter([d["lon"]],[d["lat"]],s=40,color="white",edgecolor=INK,lw=1.2,zorder=6)
        OFF=offs or {"Renault Melloussa":(8,-1,"left"),"Tanger Free Zone":(-7,6,"right"),"Tanger Automotive City":(-7,-12,"right"),"Tetouan":(8,-16,"left")}
        dx,dy,ha=OFF.get(nm,(5,-10,"left"))
        if labels: ax.annotate(nm,(d["lon"],d["lat"]),xytext=(dx,dy),textcoords="offset points",fontsize=7.5,color="#0b0b0b",ha=ha)
    for t in cfg["terminals"]:
        if extent and not (extent[0]<=t["lon"]<=extent[1] and extent[2]<=t["lat"]<=extent[3]): continue
        ax.scatter([t["lon"]],[t["lat"]],marker="s",s=26,color=BLUE,edgecolor="white",lw=0.6,zorder=5)
def fmt(ax,title=None):
    ax.set_aspect(1/np.cos(np.radians(34.7))); ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    for s in ax.spines.values(): s.set_color("#d6d5d0")
    if title: ax.set_title(title,loc="left",fontsize=10,fontweight="bold")
# ---- Fig 3 network
fig,ax=plt.subplots(figsize=(7.2,7.4)); base(ax); points(ax); fmt(ax)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=GREY,lw=1.5,label="Road (motorway–secondary)"),Line2D([],[],color="#9ec5f4",lw=1.5,label="Rail"),
    Line2D([],[],marker="*",ls="",color=INK,ms=12,label="Hub"),Line2D([],[],marker="o",ls="",mfc="white",mec=INK,ms=6,label="Hinterland destination"),
    Line2D([],[],marker="s",ls="",color=BLUE,ms=5,label="Rail–road terminal")],loc="lower right",frameon=True,fontsize=8,framealpha=0.95)
fig.tight_layout(); fig.savefig("figures/fig3_network.png",dpi=250); plt.close(fig)
# ---- Fig 4 link importance
def imp_panel(ax, extent=None):
    base(ax, extent=extent)
    e=E[E.importance_h>0].sort_values("importance_h")
    norm=LogNorm(vmin=max(1e-4,e.importance_h.min()),vmax=e.importance_h.max())
    cmap=plt.cm.Blues
    for r in e.itertuples():
        u,v=r.u,r.v
        if u not in G or v not in G: continue
        x=[G.nodes[u]["x"],G.nodes[v]["x"]]; y=[G.nodes[u]["y"],G.nodes[v]["y"]]
        c=cmap(0.35+0.65*norm(r.importance_h))
        ax.plot(x,y,color=c,lw=1.2+2.6*norm(r.importance_h),zorder=3,solid_capstyle="round",ls="-" if r.mode=="road" else (0,(2,1)))
    return norm,cmap
fig,axs=plt.subplots(1,2,figsize=(10.5,6.2),gridspec_kw={"width_ratios":[1.15,1]})
norm,cmap=imp_panel(axs[0]); points(axs[0]); fmt(axs[0],"(a) Study area")
ext=(-5.95,-5.30,35.50,35.93)
imp_panel(axs[1],extent=ext); points(axs[1],extent=ext,offs={"Renault Melloussa":(7,2,"left"),"Tanger Free Zone":(2,9,"left"),"Tanger Automotive City":(7,-10,"left"),"Tetouan":(-7,8,"right")}); axs[1].set_xlim(ext[0],ext[1]); axs[1].set_ylim(ext[2],ext[3]); fmt(axs[1],"(b) Tanger Med – Tangier – Tétouan")
sm=plt.cm.ScalarMappable(norm=LogNorm(norm.vmin,norm.vmax),cmap=plt.cm.colors.ListedColormap(cmap(np.linspace(0.35,1,256))))
cb=fig.colorbar(sm,ax=axs,shrink=0.7,pad=0.02); cb.set_label("Link importance I(e) (h, log scale)")
axs[0].legend(handles=[Line2D([],[],color=cmap(0.8),lw=2,label="Road link"),Line2D([],[],color=cmap(0.8),lw=2,ls=(0,(2,1)),label="Rail link")],loc="lower right",fontsize=8)
fig.savefig("figures/fig4_importance.png",dpi=250,bbox_inches="tight"); plt.close(fig)
# ---- Fig 5 exposure by relation
d=D.copy().sort_values("exposure_worst_delay_h")
fig,ax=plt.subplots(figsize=(7.2,4.2))
cols=[ORANGE if r=="rail-captive" else BLUE for r in d.relation]
y=np.arange(len(d))
ax.barh(y,d.exposure_worst_delay_h*60,color=cols,height=0.62,zorder=2)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(d.name,fontsize=8)
for yi,(v,k) in enumerate(zip(d.exposure_worst_delay_h*60,d.edge_disjoint_paths)):
    ax.annotate(f"{v:,.1f} min · κ={k}",(v,yi),xytext=(4,0),textcoords="offset points",va="center",fontsize=7.5,color="#52514e")
ax.set_xlabel("Worst-case delay X(d) after a single-link failure (minutes, log scale)")
ax.grid(axis="x",color="#ecebe7",zorder=0); ax.set_xlim(0.05,30000)
for s in ["top","right"]: ax.spines[s].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="Road / multimodal relation"),Patch(color=ORANGE,label="Rail-captive relation")],loc="lower right",fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig("figures/fig5_exposure.png",dpi=250); plt.close(fig)
# ---- Fig 6 governance sensitivity (S2 and S1)
dC={r.scenario.split()[0]:r.delta_connected_h for r in S.itertuples()}
Ds=np.array([0.5,1,2,3,4,5,7]); dl=np.array([0,2,4,8,12,18,24])
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for ax,sc,tt in [(axs[0],"S1","(a) S1 worst single road link"),(axs[1],"S2","(b) S2 rail-layer outage")]:
    M=np.array([[Dd*(dC[sc]+x) for Dd in Ds] for x in dl])
    im=ax.imshow(M,cmap=plt.cm.colors.ListedColormap(plt.cm.Blues(np.linspace(0.12,1,256))),origin="lower",aspect="auto",vmin=0,vmax=200)
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels([f"{x:g}" for x in Ds]); ax.set_yticks(range(len(dl))); ax.set_yticklabels(dl)
    ax.set_xlabel("Disruption duration D (days)"); ax.set_ylabel("Procedural dwell δ (h)"); ax.set_title(tt,loc="left",fontsize=10,fontweight="bold")
    for i in range(len(dl)):
        for j in range(len(Ds)):
            ax.text(j,i,f"{M[i,j]:.0f}",ha="center",va="center",fontsize=7,color="white" if M[i,j]>100 else INK)
    for (Dd,x,lab) in [(1,2,"C"),(4,12,"F")]:
        j=list(Ds).index(Dd); i=list(dl).index(x)
        ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,ec=ORANGE,lw=2))
        ax.text(j+0.32,i+0.28,lab,color=ORANGE,fontsize=8,fontweight="bold",ha="center",va="center")
cb=fig.colorbar(im,ax=axs,shrink=0.85); cb.set_label("Disruption-hours index L (h per daily unit of demand)")
fig.savefig("figures/fig6_governance.png",dpi=250,bbox_inches="tight"); plt.close(fig)
# ---- numbers
e=E.sort_values("importance_h",ascending=False); tot=e.importance_h.sum()
out={"share_rail_importance":E[E["mode"]=="rail"].importance_h.sum()/tot,
     "rail_links":int((E["mode"]=="rail").sum()),"road_links":int((E["mode"]=="road").sum())}
for q in [0.01,0.05,0.10]:
    k=max(1,int(round(len(e)*q))); out[f"top{int(q*100)}pct_n"]=k; out[f"top{int(q*100)}pct_share"]=e.importance_h.head(k).sum()/tot
r=E[E["mode"]=="road"].sort_values("importance_h",ascending=False); rt=r.importance_h.sum()
for q in [0.05,0.10]:
    k=max(1,int(round(len(r)*q))); out[f"road_top{int(q*100)}pct_share"]=r.importance_h.head(k).sum()/rt
out["phi_S3"]=dC["S3"]/(dC["S1"]+dC["S2"]); out["phi_S5"]=dC["S5"]/(dC["S4"]+dC["S2"])
out["dC"]=dC
# modal shift sensitivity: dC_S2(m)=0.1*(3m + sum(road-rail))
dd=D[D.relation=="rail-captive"]; m0=cfg["modal_shift_h"]
const=(dd.exposure_worst_delay_h.sum()-3*m0)  # worst = road+m-rail for full outage on port access
out["dC_S2_by_m"]={m:0.1*(3*m+const) for m in [6,12,24]}
out["breakeven_delta_h_S2"]=dC["S2"]; out["ratio_S2_S1"]=dC["S2"]/dC["S1"]
json.dump(out,open("numbers.json","w"),indent=2,default=float); print(json.dumps(out,indent=2,default=float))
