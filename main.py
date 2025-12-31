#################################################################
#                                       Create on 2025/12/31    #
#   WakuWakuTool                                                #
#                                                               #
#################################################################
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from typing import List
from pydantic import BaseModel
from typing import List
from datetime import date


class GraphRequest(BaseModel):
    start_date: date
    end_date: date
    products: List[str]

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ==========================
# グローバルDataFrame（Phase1）
# ==========================
df_all = pd.DataFrame()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/upload")
async def upload_csv(files: List[UploadFile] = File(...)):
    global df_all

    dfs = []
    for file in files:
        if not file.filename.lower().endswith(".csv"):
            return JSONResponse(
                status_code=400,
                content={"error": "CSVファイルのみアップロード可能です"}
            )

        df              = read_csv_with_encoding(file)
        # ===== 売上年月日 正規化 =====
        # 1) 数値・文字列混在 → 文字列
        s = df["売上年月日"]

        # 2) datetime 変換（不正は NaT）
        df["売上年月日"] = pd.to_datetime(
            s.astype("Int64", errors="ignore")
             .astype(str),
            format="%Y%m%d",
            errors="coerce"
        )

        # 3) 不正日付（NaT）行を削除
        before = len(df)
        df = df.dropna(subset=["売上年月日"])
        after = len(df)

        print(f"{file.filename}: invalid date rows removed = {before - after}")

        dfs.append(df)


    if dfs:
        df_new = pd.concat(dfs, ignore_index=True)
        df_all = df_new if df_all.empty else pd.concat(
            [df_all, df_new], ignore_index=True
        )

    return {
        "message": "CSV uploaded",
        "rows": len(df_all),
        "columns": list(df_all.columns)
    }

@app.post("/clear")
async def clear_data():
    global df_all
    df_all = pd.DataFrame()

    return JSONResponse({
        "message": "df_all cleared"
    })


@app.get("/status")
async def status():
    global df_all
    return {
        "rows": len(df_all),
        "columns": list(df_all.columns)
    }

@app.get("/filters")
async def get_filters():
    if df_all.empty:
        return {}

    return {
        "date_min": df_all["売上年月日"].min().strftime("%Y-%m-%d"),
        "date_max": df_all["売上年月日"].max().strftime("%Y-%m-%d"),
        "products": sorted(df_all["商品名"].dropna().unique().tolist()),
        "stores": sorted(df_all["店舗名"].dropna().unique().tolist())
    }


@app.post("/graph-data")
async def get_graph_data(req: GraphRequest):
    if df_all.empty:
        return JSONResponse(
            status_code=400,
            content={"error": "データが存在しません"}
        )

    # 日付範囲チェック
    min_date = df_all["売上年月日"].min().date()
    max_date = df_all["売上年月日"].max().date()

    if req.start_date > req.end_date:
        return JSONResponse(status_code=400, content={"error": "日付範囲が不正"})

    # フィルタ
    df = df_all.copy()

    df = df[
        (df["売上年月日"] >= pd.to_datetime(req.start_date)) &
        (df["売上年月日"] <= pd.to_datetime(req.end_date))
    ]

    if req.products:
        df = df[df["商品名"].isin(req.products)]

    # ===== 日別 × 店舗 集計 =====
    summary = (
        df.groupby(["売上年月日", "店舗名"])["数量"]
        .sum()
        .reset_index()
        .sort_values("売上年月日")
    )

    # --- 日曜日抽出 ---
    all_dates = pd.date_range(req.start_date, req.end_date, freq="D")
    sundays = all_dates[all_dates.weekday == 6]
    sunday_lines = []
    for d in sundays:
        sunday_lines.append({
            "type": "line",
            "x0": d.strftime("%Y-%m-%d"),
            "x1": d.strftime("%Y-%m-%d"),
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {
                "color": "red",
                "width": 1,
                "dash": "dot"
            }
        })


    # Plotly用トレース生成（店舗ごと）
    traces = []
    for store, g in summary.groupby("店舗名"):
        traces.append({
            "x": g["売上年月日"].dt.strftime("%Y-%m-%d").tolist(),
            "y": g["数量"].tolist(),
            "type": "scatter",
            "mode": "lines+markers",
            "name": store   # ← 凡例に店舗名が表示される
        })

    return {
        "traces": traces,
        "layout": {
            "title": "日別 販売個数（店舗別）",
            "xaxis": {"title": "日付"},
            "yaxis": {"title": "数量"},
            "hovermode": "x unified",
            "legend": {"title": {"text": "店舗名"}},
            "shapes": sunday_lines
        }
    }





def read_csv_with_encoding(file):
    for enc in ["utf-8", "cp932", "shift_jis"]:
        try:
            file.file.seek(0)
            return pd.read_csv(file.file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("encoding", b"", 0, 1, "unsupported encoding")








if __name__ == "__main__":
    import uvicorn

    #uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True )
    uvicorn.run("main:app", host="0.0.0.0", port=8000)