import pandas as pd
import io
output = io.BytesIO()
df = pd.DataFrame([])
try:
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
