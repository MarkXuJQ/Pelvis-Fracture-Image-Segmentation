from database.db_connection import get_connection

def update_ct_path(patient_id, ct_path):
    """更新病人的CT图像路径"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 更新 patients 表，添加 ct_image_path 字段
        cursor.execute("""
            UPDATE patients 
            SET ct_image_path = %s 
            WHERE id = %s
        """, (ct_path, patient_id))
        
        conn.commit()
        
    except Exception as e:
        print(f"更新CT路径失败: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_ct_path(patient_id):
    """获取病人的CT图像路径"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ct_image_path 
            FROM patients 
            WHERE id = %s
        """, (patient_id,))
        
        result = cursor.fetchone()
        return result[0] if result else None
        
    finally:
        cursor.close()
        conn.close() 