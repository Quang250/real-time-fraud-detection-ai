from pathlib import Path
import json
import sys
from textwrap import dedent


FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "dashboard",
    "models",
    "reports/thesis",
]


NOTEBOOK_TEMPLATE = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


FILES = {
    "notebooks/01_eda.ipynb": json.dumps(
        NOTEBOOK_TEMPLATE,
        indent=2,
        ensure_ascii=False,
    ),
    "notebooks/02_feature_engineering.ipynb": json.dumps(
        NOTEBOOK_TEMPLATE,
        indent=2,
        ensure_ascii=False,
    ),
    "notebooks/03_modeling.ipynb": json.dumps(
        NOTEBOOK_TEMPLATE,
        indent=2,
        ensure_ascii=False,
    ),
    "notebooks/04_explainability.ipynb": json.dumps(
        NOTEBOOK_TEMPLATE,
        indent=2,
        ensure_ascii=False,
    ),
    "src/data_preprocessing.py": dedent(
        '''\
        """
        Tiền xử lý dữ liệu gốc.
        Ví dụ: đọc CSV, xử lý missing values, chuẩn hóa dữ liệu.
        """

        def main():
            print("Run data preprocessing here.")

        if __name__ == "__main__":
            main()
        '''
    ),
    "src/feature_engineering.py": dedent(
        '''\
        """
        Tạo đặc trưng mới cho mô hình.
        Ví dụ: scaling, interaction features, PCA, encoding.
        """

        def main():
            print("Run feature engineering here.")

        if __name__ == "__main__":
            main()
        '''
    ),
    "src/train_model.py": dedent(
        '''\
        """
        Huấn luyện mô hình.
        Ví dụ: Logistic Regression, XGBoost, LightGBM.
        """

        def main():
            print("Run model training here.")

        if __name__ == "__main__":
            main()
        '''
    ),
    "src/evaluate_model.py": dedent(
        '''\
        """
        Đánh giá mô hình.
        Ví dụ: Accuracy, Precision, Recall, F1, ROC-AUC.
        """

        def main():
            print("Run evaluation here.")

        if __name__ == "__main__":
            main()
        '''
    ),
    "src/explainability.py": dedent(
        '''\
        """
        Giải thích mô hình bằng SHAP / LIME.
        """

        def main():
            print("Run explainability here.")

        if __name__ == "__main__":
            main()
        '''
    ),
    "src/inference_api.py": dedent(
        '''\
        """
        API suy luận cho mô hình.
        Ví dụ: FastAPI endpoint /predict
        """

        def main():
            print("Run inference API here.")

        if __name__ == "__main__":
            main()
        '''
    ),
    "dashboard/streamlit_app.py": dedent(
        '''\
        import streamlit as st

        st.set_page_config(page_title="Fraud Detection XAI Dashboard", layout="wide")
        st.title("Fraud Detection XAI Dashboard")
        st.write("Đây là dashboard demo ban đầu.")
        '''
    ),
    "requirements.txt": dedent(
        """\
        pandas
        numpy
        scikit-learn
        matplotlib
        seaborn
        jupyter
        notebook
        xgboost
        lightgbm
        shap
        lime
        fastapi
        uvicorn
        streamlit
        """
    ),
    "README.md": dedent(
        """\
        # Fraud Detection XAI

        ## Project structure
        - `data/raw`: dữ liệu gốc
        - `data/processed`: dữ liệu đã xử lý
        - `notebooks`: notebook theo từng giai đoạn
        - `src`: mã nguồn chính
        - `dashboard`: ứng dụng Streamlit
        - `models`: nơi lưu model đã train
        - `reports/thesis`: tài liệu báo cáo / khóa luận

        ## Suggested workflow
        1. Đặt dữ liệu gốc vào `data/raw`
        2. Chạy notebook `01_eda.ipynb`
        3. Viết code xử lý trong `src/data_preprocessing.py`
        4. Huấn luyện trong `src/train_model.py`
        5. Giải thích mô hình trong `src/explainability.py`
        6. Tạo dashboard bằng Streamlit

        ## Run dashboard
        ```bash
        streamlit run dashboard/streamlit_app.py
        ```
        """
    ),
}


def create_project_structure(base_dir: Path) -> None:
    base_dir = base_dir.resolve()

    for folder in FOLDERS:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)

    for relative_path, content in FILES.items():
        file_path = base_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    print(f"Project structure created successfully at: {base_dir}")


def parse_base_dir() -> Path:
    if len(sys.argv) > 2:
        script_name = Path(sys.argv[0]).name
        raise SystemExit(f"Usage: python {script_name} [target_directory]")

    return Path(sys.argv[1]) if len(sys.argv) == 2 else Path.cwd()


if __name__ == "__main__":
    create_project_structure(parse_base_dir())
