#!/usr/bin/env python3
"""Fix Journey column references in app.py"""

import re

# Read the app.py file
with open('/Users/leonida/Documents/automobile_claims/Automobile/app.py', 'r') as f:
    content = f.read()

# Replace all the old journey references with the new ones
replacements = [
    (r"'Protect'", "'NEW_CUSTOMER'"),
    (r"'Grow'", "'DEVELOPING'"), 
    (r"'Rescue'", "'ESTABLISHED'"),
    (r"'Monitor'", "'LOYAL_VETERAN'"),
    (r"'Develop'", "'DEVELOPING'"),
    (r'"PROTECT"', '"NEW CUSTOMERS"'),
    (r'"DEVELOP"', '"DEVELOPING"'),
    (r'"RESCUE"', '"ESTABLISHED"'),
    (r'"MONITOR"', '"LOYAL VETERANS"'),
    (r'"GROW"', '"DEVELOPING"'),
    (r'get\(\'Protect\'', "get('NEW_CUSTOMER'"),
    (r'get\(\'Grow\'', "get('DEVELOPING'"),
    (r'get\(\'Rescue\'', "get('ESTABLISHED'"),
    (r'get\(\'Monitor\'', "get('LOYAL_VETERAN'"),
    (r'get\(\'Develop\'', "get('DEVELOPING'"),
    (r'Journey Quadrant', 'Customer Journey Stage'),
    (r'Journey:', 'Journey Stage:'),
]

for old, new in replacements:
    content = re.sub(old, new, content)

# Write the fixed content back
with open('/Users/leonida/Documents/automobile_claims/Automobile/app.py', 'w') as f:
    f.write(content)

print("✅ Fixed Journey column references in app.py")