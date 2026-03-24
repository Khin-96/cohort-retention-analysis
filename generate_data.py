import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)

def generate_synthetic_data(num_users=1000, start_date='2023-01-01', end_date='2024-03-01'):
    users_data = []
    orders_data = []
    payments_data = []

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end_dt - start_dt).days

    for i in range(num_users):
        user_id = f"user_{i:04d}"
        signup_offset = random.randint(0, date_range - 60) # Signup at least 2 months before end
        signup_date = start_dt + timedelta(days=signup_offset)
        
        users_data.append({
            'user_id': user_id,
            'signup_date': signup_date,
            'user_type': random.choice(['Basic', 'Premium', 'Corporate'])
        })

        # Generate orders for this user
        num_orders = random.randint(1, 40 if random.random() > 0.3 else 5) # Power law distribution simulation
        
        # Ensure first order date equals signup date for simplicity
        current_order_date = signup_date
        
        for j in range(num_orders):
            if current_order_date > end_dt:
                break
                
            order_id = f"ord_{user_id}_{j:02d}"
            order_value = round(random.uniform(10.0, 150.0), 2)
            
            orders_data.append({
                'order_id': order_id,
                'user_id': user_id,
                'order_date': current_order_date,
                'order_value': order_value
            })
            
            # Payment record
            payments_data.append({
                'payment_id': f"pay_{order_id}",
                'order_id': order_id,
                'payment_status': 'Success' if random.random() > 0.05 else 'Failed',
                'payment_method': random.choice(['Credit Card', 'Apple Pay', 'Google Pay', 'Cash'])
            })
            
            # Gap between orders
            gap = random.randint(1, 90)
            current_order_date += timedelta(days=gap)

    users_df = pd.DataFrame(users_data)
    orders_df = pd.DataFrame(orders_data)
    payments_df = pd.DataFrame(payments_data)

    return users_df, orders_df, payments_df

if __name__ == "__main__":
    print("Generating synthetic Glovo-style data...")
    users, orders, payments = generate_synthetic_data()
    
    users.to_csv('users.csv', index=False)
    orders.to_csv('orders.csv', index=False)
    payments.to_csv('payments.csv', index=False)
    
    print(f"Generated {len(users)} users, {len(orders)} orders, and {len(payments)} payments.")
    print("Files saved: users.csv, orders.csv, payments.csv")
