#!/usr/bin/env bash
# Sửa lỗi "XGBoost Library could not be loaded ... libomp.dylib" trên macOS (Apple Silicon)
# khi KHÔNG có Homebrew. Mượn libomp.dylib mà scikit-learn đóng gói sẵn, đặt cạnh
# libxgboost.dylib và thêm rpath @loader_path. Chạy lại sau mỗi lần dựng lại venv / cài lại xgboost.
#
# Cách dùng:  bash fix_xgboost_openmp.sh
set -e

VENV="${1:-venv}"
SITE="$VENV/lib/python3.9/site-packages"
XGB_LIB="$SITE/xgboost/lib"
OMP_SRC="$SITE/sklearn/.dylibs/libomp.dylib"

if [ ! -f "$XGB_LIB/libxgboost.dylib" ]; then
  echo "Không thấy $XGB_LIB/libxgboost.dylib — kiểm tra lại đường dẫn venv."; exit 1
fi
if [ ! -f "$OMP_SRC" ]; then
  echo "Không thấy libomp.dylib của scikit-learn ($OMP_SRC). Hãy 'pip install scikit-learn' trước."; exit 1
fi

cp -f "$OMP_SRC" "$XGB_LIB/libomp.dylib"
# Thêm @loader_path nếu chưa có
if ! otool -l "$XGB_LIB/libxgboost.dylib" | grep -q "@loader_path"; then
  install_name_tool -add_rpath @loader_path "$XGB_LIB/libxgboost.dylib"
fi
# Ký lại ad-hoc để arm64 chịu nạp
codesign --force --sign - "$XGB_LIB/libxgboost.dylib"
codesign --force --sign - "$XGB_LIB/libomp.dylib"

"$VENV/bin/python" -c "import xgboost; print('OK XGBoost', xgboost.__version__)"
