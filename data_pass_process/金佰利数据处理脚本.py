"""
金佰利全国数据处理脚本
功能：将数据透视转换为行=月份，列=门店名称的格式，保留门店城市和客户信息，缺失值用0填充
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_jinbaili_data(file_path, pivot_value_col="达成", output_dir="./output"):
    """
    处理金佰利数据的主函数
    
    参数:
    file_path: Excel文件路径
    pivot_value_col: 要透视的数值列名，默认是"达成"（实际销售数据）
    output_dir: 输出目录
    
    返回:
    dict: 包含处理后的数据字典
    """
    
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. 读取数据
    print(f"正在读取数据: {file_path}")
    try:
        # 跳过第一行，使用第二行作为列名
        df = pd.read_excel(file_path, header=1)
        print(f"数据读取成功，数据形状: {df.shape}")
    except Exception as e:
        print(f"数据读取失败: {e}")
        raise
    
    # 3. 数据预处理
    print("\n开始数据预处理...")
    
    # 3.1 检查必要列是否存在
    required_columns = ["门店名称", "月份", "地城市", "大区", "小区", pivot_value_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    # 3.2 处理重复的门店-月份组合（取平均值）
    duplicate_check = df.groupby(["门店名称", "月份"]).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    if len(duplicates) > 0:
        print(f"发现 {len(duplicates)} 个重复的门店-月份组合，将按平均值合并")
        
        # 对于数值列取平均，对于分类列取第一个值
        agg_dict = {}
        for col in df.columns:
            if col in [pivot_value_col, "目标", "去年同期销售额", "考勤次数合计"]:
                agg_dict[col] = "mean"  # 数值列取平均
            elif col not in ["门店名称", "月份"]:
                agg_dict[col] = "first"  # 其他列取第一个值
        
        # 按门店名称和月份分组聚合
        df = df.groupby(["门店名称", "月份"], as_index=False).agg(agg_dict)
        print(f"重复数据处理完成，处理后数据形状: {df.shape}")
    
    # 3.3 创建门店信息表（包含城市、大区等信息）
    store_info = df[["门店名称", "地城市", "大区", "小区", "门店编码"]].drop_duplicates("门店名称")
    store_info = store_info.reset_index(drop=True)
    print(f"门店信息表创建完成，包含 {len(store_info)} 个门店")
    
    # 4. 数据透视
    print(f"\n开始数据透视，透视指标: {pivot_value_col}")
    
    # 创建透视表：行=月份，列=门店名称，值=透视指标，缺失值填充0
    pivot_table = df.pivot_table(
        index="月份",
        columns="门店名称",
        values=pivot_value_col,
        aggfunc="sum",  # 确保同一月份同一门店只有一个值
        fill_value=0
    )
    
    print(f"透视表创建完成")
    print(f"透视表形状: {pivot_table.shape}")
    print(f"时间范围: {sorted(pivot_table.index.tolist())}")
    print(f"门店数量: {len(pivot_table.columns)}")
    
    # 5. 月份排序（确保按时间顺序排列）
    # 定义月份排序函数
    def sort_months(month_list):
        # 提取年份和月份信息进行排序
        def month_key(month_str):
            year = int(month_str[1:3])  # 从Y22-01提取22
            month = int(month_str[-2:])  # 从Y22-01提取01
            return (year, month)
        
        return sorted(month_list, key=month_key)
    
    # 对透视表的索引进行排序
    sorted_months = sort_months(pivot_table.index.tolist())
    pivot_table = pivot_table.reindex(sorted_months)
    print(f"月份排序完成，排序后的月份: {sorted_months}")
    
    # 6. 保存结果
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"金佰利数据处理结果_{current_time}"
    
    # 6.1 保存透视表
    pivot_output_path = os.path.join(output_dir, f"{output_prefix}_透视表.xlsx")
    pivot_table.to_excel(pivot_output_path)
    print(f"\n透视表已保存至: {pivot_output_path}")
    
    # 6.2 保存门店信息表
    store_info_output_path = os.path.join(output_dir, f"{output_prefix}_门店信息表.xlsx")
    store_info.to_excel(store_info_output_path, index=False)
    print(f"门店信息表已保存至: {store_info_output_path}")
    
    # 6.3 保存完整的处理后数据（可选）
    full_data_output_path = os.path.join(output_dir, f"{output_prefix}_处理后完整数据.xlsx")
    with pd.ExcelWriter(full_data_output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="处理后数据", index=False)
        pivot_table.to_excel(writer, sheet_name="透视表")
        store_info.to_excel(writer, sheet_name="门店信息")
    print(f"完整处理结果已保存至: {full_data_output_path}")
    
    # 7. 数据质量报告
    print("\n" + "="*50)
    print("数据处理质量报告")
    print("="*50)
    print(f"原始数据总行数: {df.shape[0]}")
    print(f"门店总数: {len(store_info)}")
    print(f"时间跨度: {len(sorted_months)} 个月")
    print(f"透视表总行数: {pivot_table.shape[0]}")
    print(f"透视表总列数: {pivot_table.shape[1]}")
    print(f"总数据单元格数: {pivot_table.shape[0] * pivot_table.shape[1]}")
    print(f"零值单元格比例: {round((pivot_table == 0).sum().sum() / (pivot_table.shape[0] * pivot_table.shape[1]) * 100, 2)}%")
    
    # 8. 返回处理结果
    result = {
        "pivot_table": pivot_table,
        "store_info": store_info,
        "processed_data": df,
        "output_paths": {
            "pivot_table": pivot_output_path,
            "store_info": store_info_output_path,
            "full_data": full_data_output_path
        }
    }
    
    print("\n数据处理完成！")
    return result

# ------------------------------------------------------------------------------
# 主程序执行
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "全国数据（修正）金佰利.xlsx"  # 输入文件路径
    PIVOT_COLUMN = "达成"  # 可选择的指标：达成、目标、去年同期销售额、考勤次数合计等
    OUTPUT_DIR = "金佰利数据处理结果"  # 输出目录
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 '{INPUT_FILE}' 不存在！")
        print("请确保输入文件路径正确。")
    else:
        # 执行数据处理
        try:
            processing_result = process_jinbaili_data(
                file_path=INPUT_FILE,
                pivot_value_col=PIVOT_COLUMN,
                output_dir=OUTPUT_DIR
            )
            
            # 显示处理结果预览
            print("\n" + "="*30)
            print("处理结果预览")
            print("="*30)
            print("透视表前5行前5列:")
            print(processing_result["pivot_table"].iloc[:5, :5])
            
            print("\n门店信息表前5行:")
            print(processing_result["store_info"].head())
            
        except Exception as e:
            print(f"数据处理过程中发生错误: {e}")
            print("请检查数据格式或联系技术支持。")
