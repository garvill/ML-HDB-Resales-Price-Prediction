from ML import predict_price

flat_info = {
    'floor_area_sqm': 100,
    'flat_type': '5-ROOM',
    'storey_range': '04 TO 06',
    'lease_remaining': 89,
    'town': 'CHOA CHU KANG',
    'year': 2022
}

price = predict_price(flat_info)
print(f"Predicted resale price: ${price[0]:,.2f}")
