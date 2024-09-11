from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import tensorflow as tf

# 连接到 Milvus 服务器
connections.connect("default", host='localhost', port='19530')

# 定义集合名称和参数
collection_name = 'test_collection_bf16_new'
dim = 1024  # 向量维度

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.BFLOAT16_VECTOR, dim=dim)
]

# 创建集合
schema = CollectionSchema(fields, description="Test Collection")
collection = Collection(name=collection_name, schema=schema)

# 生成示例数据
num_entities = 16000
ids = np.arange(num_entities).tolist()
#vectors = np.random.random((num_entities, dim)).astype(np.float32).tolist()

vectors = (np.random.random((num_entities, dim)).astype(np.float16)*2-1)
vectors = tf.cast(vectors, dtype=tf.bfloat16).numpy()


# 批量插入数据
entities = [
    ids,  # 直接插入 ID 列表
    vectors  # 直接插入向量列表
]

insert_result = collection.insert(entities)

# 创建索引
index_params = {
    "metric_type": "IP",
    "index_type": "FLAT"
}

collection.create_index(field_name="vector", index_params=index_params)

# 确保数据已插入
#collection.load()
collection.flush()

# 检查集合中的数据数量
print(f"Number of entities in collection: {collection.num_entities}")

# 断开连接
connections.disconnect("default")
