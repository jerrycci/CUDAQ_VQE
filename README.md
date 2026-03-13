# 1. 安裝依賴
pip install cudaq pyscf openfermion openfermionpyscf websockets scipy numpy

# 2. 啟動後端（會自動偵測 RTX 5080）
python vqe_server.py

# 輸出應顯示：
#   GPU 0  : NVIDIA GeForce RTX 5080  16384MB
#   WebSocket: ws://0.0.0.0:8765

# 3. 瀏覽器開啟 cudaq_vqe_realtime.html
#    輸入 ws://your-server-ip:8765 連接
