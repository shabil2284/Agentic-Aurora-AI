import re
import sys

with open("app.py", "r") as f:
    text = f.read()

start_str = '    if st.session_state.df is not None:\n        st.markdown("<br><h3>🏗️ Master Data & Schema</h3>", unsafe_allow_html=True)'
end_str = '        st.info("System idle. Process a document to view data.")'

if start_str not in text:
    print("Could not find start str")
    sys.exit(1)
if end_str not in text:
    print("Could not find end str")
    sys.exit(1)

start_idx = text.find(start_str)
end_idx = text.find(end_str) + len(end_str)

replacement = """    if st.session_state.df is not None:
        st.markdown("<br><h3>🏗️ Master Data & Schema</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### CSV Editor")

            col1, col2 = st.columns([6, 2])
            with col1:
                st.session_state.auto_sync_enabled = st.toggle("Auto Sync", value=st.session_state.auto_sync_enabled, key="autosync")
            with col2:
                sync_now = st.button("🔄 Sync Now", use_container_width=True)

            if st.session_state.auto_sync_enabled:
                st.success("🟢 Auto Sync Active")
            else:
                st.warning("🟡 Manual Sync Mode")

            if sync_now:
                execute_master_sync("MANUAL_BUTTON", force=True)
                st.rerun()

            # Render interactive DataGrid bound strictly to st.session_state.df
            edited_df = st.data_editor(
                st.session_state.df,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key=f"csv_editor_{st.session_state.get('grid_key_counter', 0)}",
                height=720
            )

            # Master Sync Hook: df -> SQL + JSON
            _changed = (
                st.session_state.df.shape != edited_df.shape or
                st.session_state.df.to_csv(index=False) != edited_df.to_csv(index=False)
            )
            if _changed and not st.session_state.is_syncing:
                st.session_state.df = edited_df.copy()
                st.session_state.csv_df = st.session_state.df
                st.session_state.json_data = st.session_state.df.to_dict(orient="records")
                if "grid_key_counter" not in st.session_state:
                    st.session_state.grid_key_counter = 0
                st.session_state.grid_key_counter += 1
                if st.session_state.auto_sync_enabled:
                    execute_master_sync("CSV_DATAGRID_EDIT")
                st.rerun()
                
        with c2:
            st.markdown("#### Auto Generated Schema Viewer")
            current_schema = [
                {
                    "Field Name": col, 
                    "Data Type": "string" if st.session_state.df[col].dtype == "object" else str(st.session_state.df[col].dtype),
                    "Editable": True
                }
                for col in st.session_state.df.columns
            ]
            
            edited_schema_df = st.data_editor(
                current_schema,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key=f"schema_editor_{st.session_state.get('grid_key_counter', 0)}",
                height=250
            )

            edited_cols = []
            for row in edited_schema_df:
                c_name = str(row.get("Field Name", "")).strip()
                if not c_name: c_name = f"Unnamed_{len(edited_cols)}"
                edited_cols.append(c_name)

            current_cols = list(st.session_state.df.columns)
            _schema_changed = False
            
            if len(edited_cols) == len(current_cols) and edited_cols != current_cols:
                rename_dict = {old: new for old, new in zip(current_cols, edited_cols) if old != new}
                st.session_state.df = st.session_state.df.rename(columns=rename_dict)
                _schema_changed = True
            elif len(edited_cols) < len(current_cols):
                valid_cols = [c for c in edited_cols if c in current_cols]
                st.session_state.df = st.session_state.df[valid_cols]
                _schema_changed = True
            elif len(edited_cols) > len(current_cols):
                for c in edited_cols:
                    if c not in current_cols:
                        st.session_state.df[c] = None
                valid_ordered_cols = [c for c in edited_cols if c in st.session_state.df.columns]
                st.session_state.df = st.session_state.df[valid_ordered_cols]
                _schema_changed = True

            if not _schema_changed:
                for old_row, new_row in zip(current_schema, edited_schema_df):
                    if old_row.get("Data Type") != new_row.get("Data Type"):
                        col_name = new_row.get("Field Name")
                        new_type = str(new_row.get("Data Type")).lower()
                        try:
                            if "int" in new_type: st.session_state.df[col_name] = __import__('pandas').to_numeric(st.session_state.df[col_name], errors='coerce').fillna(0).astype('Int64')
                            elif "float" in new_type: st.session_state.df[col_name] = __import__('pandas').to_numeric(st.session_state.df[col_name], errors='coerce')
                            elif "str" in new_type or "object" in new_type: st.session_state.df[col_name] = st.session_state.df[col_name].astype(str)
                            elif "bool" in new_type: st.session_state.df[col_name] = st.session_state.df[col_name].astype(bool)
                            _schema_changed = True
                        except Exception as e:
                            print(f"[SchemaSync] Skipped dtype conversion error: {e}")

            if _schema_changed:
                st.session_state.csv_df = st.session_state.df
                st.session_state.json_data = st.session_state.df.to_dict(orient="records")
                if "grid_key_counter" not in st.session_state:
                    st.session_state.grid_key_counter = 0
                st.session_state.grid_key_counter += 1
                if st.session_state.auto_sync_enabled:
                    execute_master_sync("SCHEMA_EDIT")
                st.rerun()

            st.markdown("<br>#### JSON Viewer", unsafe_allow_html=True)
            with st.container(height=350):
                st.json(st.session_state.json_data)
        
        st.divider()
        st.markdown("#### Export Data")
        exc_col, csv_col, json_col = st.columns(3)
        with exc_col:
            export_df = st.session_state.csv_df.copy()
            import io
            output = io.BytesIO()
            if not export_df.empty:
                with __import__('pandas').ExcelWriter(output, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False)
                st.download_button("📥 Export Excel", data=output.getvalue(), file_name="extraction.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            else:
                st.download_button("📥 Export Excel", data=b"", file_name="empty.xlsx", disabled=True, use_container_width=True)
        with csv_col:
            if not export_df.empty:
                st.download_button("📥 Export CSV", data=export_df.to_csv(index=False).encode("utf-8"), file_name="extraction.csv", mime="text/csv", use_container_width=True)
            else:
                st.download_button("📥 Export CSV", data=b"", file_name="empty.csv", disabled=True, use_container_width=True)
        with json_col:
            _src_df = st.session_state.df if st.session_state.df is not None else st.session_state.csv_df
            if _src_df is not None and not _src_df.empty:
                import json
                _json_str = json.dumps(_src_df.to_dict(orient="records"), indent=2, ensure_ascii=False)
                st.download_button("📥 Export JSON", data=_json_str, file_name="extraction.json", mime="application/json", key="json_download_btn", use_container_width=True)
            else:
                st.download_button("📥 Export JSON", data=b"", file_name="empty.json", disabled=True, use_container_width=True)
    else:
        st.info("System idle. Process a document to view data.")"""

with open("app.py", "w") as f:
    f.write(text[:start_idx] + replacement + text[end_idx:])

print("Alignment patch matched and applied successfully!")
