```python
import base64

# Dari response
for name, vec_data in result['vectors'].items():
    with open(vec_data['filename'], 'wb') as f:
        f.write(base64.b64decode(vec_data['data']))
```