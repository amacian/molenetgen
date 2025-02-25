import os
import pandas as pd
import networkconstants as nc
import numpy as np


def process_xlsx_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

    if not files:
        print("Did not find any .xlsx file in the directory.")
        return

    temp_df = None
    links_df = None
    for file in files:
        file_path = os.path.join(directory, file)

        try:
            # Leer la primera hoja del archivo Excel
            df = pd.read_excel(file_path, sheet_name=nc.NODES_EXCEL_NAME)
            dfl = pd.read_excel(file_path, sheet_name=nc.LINKS_EXCEL_NAME)
            if links_df is None:
                links_df = dfl
            else:
                links_df = pd.concat([dfl, links_df])
            if temp_df is None:
                temp_df = df
            else:
                temp_df = temp_df.merge(df, "outer", on=[nc.XLS_NODE_NAME], suffixes=('', '_mer2'))

                new_df = pd.DataFrame({
                        nc.XLS_NODE_NAME: temp_df[nc.XLS_NODE_NAME],
                        nc.XLS_CO_TYPE: np.where(temp_df[nc.XLS_CO_TYPE] == nc.NATIONAL_CO_CODE,
                                                 nc.NATIONAL_CO_CODE,
                                                 np.where(temp_df[nc.XLS_CO_TYPE+"_mer2"].isna(),
                                                          temp_df[nc.XLS_CO_TYPE],
                                                          temp_df[nc.XLS_CO_TYPE+"_mer2"])),
                        nc.XLS_REF_RCO: np.where(temp_df[nc.XLS_REF_RCO].isna(), temp_df[nc.XLS_REF_RCO+"_mer2"],
                                        temp_df[nc.XLS_REF_RCO]),
                        nc.XLS_REF_NCO: np.where(temp_df[nc.XLS_REF_NCO].isna(), temp_df[nc.XLS_REF_NCO+"_mer2"],
                                        temp_df[nc.XLS_REF_NCO]),
                        nc.XLS_HOUSE_H: np.where(temp_df[nc.XLS_HOUSE_H].isna(), temp_df[nc.XLS_HOUSE_H+"_mer2"],
                                        temp_df[nc.XLS_HOUSE_H]),
                        nc.XLS_MACRO_C: np.where(temp_df[nc.XLS_MACRO_C].isna(), temp_df[nc.XLS_MACRO_C+"_mer2"],
                                        temp_df[nc.XLS_MACRO_C]),
                        nc.XLS_SMALL_C: np.where(temp_df[nc.XLS_SMALL_C].isna(), temp_df[nc.XLS_SMALL_C+"_mer2"],
                                        temp_df[nc.XLS_SMALL_C]),
                        nc.XLS_TWIN_RCO: np.where(temp_df[nc.XLS_TWIN_RCO].isna(), temp_df[nc.XLS_TWIN_RCO+"_mer2"],
                                        temp_df[nc.XLS_TWIN_RCO]),
                        nc.XLS_TWIN_NCO: np.where(temp_df[nc.XLS_TWIN_NCO].isna(), temp_df[nc.XLS_TWIN_NCO+"_mer2"],
                                        temp_df[nc.XLS_TWIN_NCO]),
                        nc.XLS_CLUSTER: np.where(temp_df[nc.XLS_CLUSTER].isna(), temp_df[nc.XLS_CLUSTER+"_mer2"],
                                        temp_df[nc.XLS_CLUSTER]),
                        nc.XLS_X_BACK: np.where(temp_df[nc.XLS_X_BACK].isna(), temp_df[nc.XLS_X_BACK+"_mer2"],
                                        temp_df[nc.XLS_X_BACK]),
                        nc.XLS_Y_BACK: np.where(temp_df[nc.XLS_Y_BACK].isna(), temp_df[nc.XLS_Y_BACK+"_mer2"],
                                        temp_df[nc.XLS_Y_BACK]),
                        nc.XLS_X_MCORE: np.where(temp_df[nc.XLS_X_MCORE].isna(), temp_df[nc.XLS_X_MCORE+"_mer2"],
                                        temp_df[nc.XLS_X_MCORE]),
                        nc.XLS_Y_MCORE: np.where(temp_df[nc.XLS_Y_MCORE].isna(), temp_df[nc.XLS_Y_MCORE+"_mer2"],
                                        temp_df[nc.XLS_Y_MCORE])

                })
                temp_df = new_df
            # dataframes[file] = df
            print(f"Processed: {file}")
            print()
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return temp_df, links_df


if __name__ == '__main__':
    directory_path = "path_to_files/"  # change to the name of the actual directory
    xls_node_data, xls_link_data = process_xlsx_files(directory_path)
    with pd.ExcelWriter(directory_path + "/topology_file.xlsx", engine='openpyxl') as writer:
        xls_node_data.to_excel(writer, sheet_name=nc.NODES_EXCEL_NAME, index=False)
        xls_link_data.to_excel(writer, sheet_name=nc.LINKS_EXCEL_NAME, index=False)
