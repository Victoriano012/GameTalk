
############ GEMINI GENERATED CODE ############
# https://g.co/gemini/share/e613404b465f


import random

def calculate_demand_den(cost, max_price):
  """Calculates demand_den ensuring demand at cost is between 10 and 1000."""
  if max_price <= cost:
    # Avoid division by zero or negative demand_den if max_price is not greater than cost
    # Set a default demand_den, although this case shouldn't happen with proper inputs
    print(f"Warning: Max price ({max_price}) is not greater than cost ({cost}). Setting default demand_den.")
    return 1
  # Calculate the range for demand_den
  min_demand_at_cost = 10
  max_demand_at_cost = 1000
  # Choose a random demand level within the desired range when price equals cost
  demand_at_cost = random.randint(min_demand_at_cost, max_demand_at_cost)
  # Calculate demand_den based on this chosen demand level
  demand_den = (max_price - cost) / demand_at_cost
  # Ensure demand_den is at least a small positive number to avoid issues
  return max(0.01, demand_den)

products_list = [
  # Electronics (20)
  {'products': ['Laptops'], 'cost': [500], 'max_price_with_demand': [2000], 'demand_den': [calculate_demand_den(500, 2000)]},
  {'products': ['Smartphones'], 'cost': [300], 'max_price_with_demand': [1500], 'demand_den': [calculate_demand_den(300, 1500)]},
  {'products': ['Tablets'], 'cost': [150], 'max_price_with_demand': [800], 'demand_den': [calculate_demand_den(150, 800)]},
  {'products': ['Televisions'], 'cost': [400], 'max_price_with_demand': [3000], 'demand_den': [calculate_demand_den(400, 3000)]},
  {'products': ['Headphones'], 'cost': [50], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(50, 400)]},
  {'products': ['Smartwatches'], 'cost': [100], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(100, 500)]},
  {'products': ['Digital Cameras'], 'cost': [250], 'max_price_with_demand': [1200], 'demand_den': [calculate_demand_den(250, 1200)]},
  {'products': ['Drones'], 'cost': [200], 'max_price_with_demand': [1800], 'demand_den': [calculate_demand_den(200, 1800)]},
  {'products': ['Gaming Consoles'], 'cost': [350], 'max_price_with_demand': [700], 'demand_den': [calculate_demand_den(350, 700)]},
  {'products': ['VR Headsets'], 'cost': [300], 'max_price_with_demand': [1000], 'demand_den': [calculate_demand_den(300, 1000)]},
  {'products': ['Bluetooth Speakers'], 'cost': [50], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(50, 300)]},
  {'products': ['E-readers'], 'cost': [80], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(80, 250)]},
  {'products': ['Computer Monitors'], 'cost': [150], 'max_price_with_demand': [600], 'demand_den': [calculate_demand_den(150, 600)]},
  {'products': ['Keyboards'], 'cost': [50], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(50, 200)]},
  {'products': ['Computer Mice'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},
  {'products': ['Webcams'], 'cost': [50], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(50, 180)]},
  {'products': ['Wireless Routers'], 'cost': [60], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(60, 300)]},
  {'products': ['External Hard Drives'], 'cost': [70], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(70, 250)]},
  {'products': ['Printers'], 'cost': [100], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(100, 500)]},
  {'products': ['Scanners'], 'cost': [80], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(80, 400)]},

  # Clothing & Accessories (20)
  {'products': ['T-shirts'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost to meet >= 50
  {'products': ['Jeans'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]}, # Adjusted cost
  {'products': ['Dresses'], 'cost': [60], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(60, 250)]},
  {'products': ['Sweaters'], 'cost': [55], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(55, 180)]},
  {'products': ['Jackets'], 'cost': [80], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(80, 400)]},
  {'products': ['Running Shoes'], 'cost': [60], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(60, 200)]},
  {'products': ['Leather Boots'], 'cost': [90], 'max_price_with_demand': [350], 'demand_den': [calculate_demand_den(90, 350)]},
  {'products': ['Sandals'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Adjusted cost
  {'products': ['Baseball Caps'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Winter Scarves'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]}, # Adjusted cost
  {'products': ['Leather Gloves'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]}, # Adjusted cost
  {'products': ['Leather Belts'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Adjusted cost
  {'products': ['Sunglasses'], 'cost': [50], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(50, 250)]},
  {'products': ['Wristwatches'], 'cost': [70], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(70, 500)]},
  {'products': ['Backpacks'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},
  {'products': ['Handbags'], 'cost': [75], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(75, 400)]},
  {'products': ['Wallets'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]}, # Adjusted cost
  {'products': ['Cotton Socks (Multi-packs)'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Boxer Briefs (Multi-packs)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Silk Ties'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Adjusted cost

  # Home & Kitchen (20)
  {'products': ['Sofas'], 'cost': [400], 'max_price_with_demand': [1500], 'demand_den': [calculate_demand_den(400, 1500)]},
  {'products': ['Dining Chairs'], 'cost': [70], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(70, 250)]},
  {'products': ['Coffee Tables'], 'cost': [100], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(100, 400)]},
  {'products': ['Queen Beds'], 'cost': [300], 'max_price_with_demand': [1000], 'demand_den': [calculate_demand_den(300, 1000)]},
  {'products': ['Memory Foam Mattresses'], 'cost': [350], 'max_price_with_demand': [1200], 'demand_den': [calculate_demand_den(350, 1200)]},
  {'products': ['Floor Lamps'], 'cost': [60], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(60, 200)]},
  {'products': ['Area Rugs'], 'cost': [100], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(100, 500)]},
  {'products': ['Blackout Curtains'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},
  {'products': ['Wall Mirrors'], 'cost': [50], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(50, 200)]},
  {'products': ['Wall Clocks'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Adjusted cost
  {'products': ['Cookware Sets'], 'cost': [120], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(120, 400)]},
  {'products': ['Chef Knives'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},
  {'products': ['High-Speed Blenders'], 'cost': [150], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(150, 500)]},
  {'products': ['Drip Coffee Makers'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]},
  {'products': ['Toasters (4-slice)'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Microwave Ovens'], 'cost': [80], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(80, 250)]},
  {'products': ['Refrigerators'], 'cost': [600], 'max_price_with_demand': [2000], 'demand_den': [calculate_demand_den(600, 2000)]},
  {'products': ['Dishwashers'], 'cost': [300], 'max_price_with_demand': [800], 'demand_den': [calculate_demand_den(300, 800)]},
  {'products': ['Washing Machines'], 'cost': [400], 'max_price_with_demand': [1000], 'demand_den': [calculate_demand_den(400, 1000)]},
  {'products': ['Vacuum Cleaners'], 'cost': [100], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(100, 500)]},

  # Food & Grocery (High Volume - Adjusted Costs/Prices/Demand) - Costs are likely per larger unit/case
  {'products': ['Organic Apples (Cases)'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Imported Cheeses (Wheels)'], 'cost': [80], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(80, 200)]},
  {'products': ['Premium Olive Oils (Cases)'], 'cost': [60], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(60, 150)]},
  {'products': ['Artisan Breads (Bulk Order)'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]},
  {'products': ['Gourmet Coffee Beans (Bulk Bags)'], 'cost': [70], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(70, 180)]},
  {'products': ['Specialty Teas (Assortment Cases)'], 'cost': [55], 'max_price_with_demand': [130], 'demand_den': [calculate_demand_den(55, 130)]},
  {'products': ['Craft Chocolate Bars (Display Boxes)'], 'cost': [50], 'max_price_with_demand': [110], 'demand_den': [calculate_demand_den(50, 110)]},
  {'products': ['Dried Fruits & Nuts (Bulk)'], 'cost': [65], 'max_price_with_demand': [140], 'demand_den': [calculate_demand_den(65, 140)]},
  {'products': ['Imported Pasta (Cases)'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Canned San Marzano Tomatoes (Cases)'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]},
  {'products': ['Organic Chicken Breasts (Bulk)'], 'cost': [100], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(100, 250)]},
  {'products': ['Grass-Fed Ground Beef (Bulk)'], 'cost': [90], 'max_price_with_demand': [220], 'demand_den': [calculate_demand_den(90, 220)]},
  {'products': ['Wild-Caught Salmon Fillets (Bulk)'], 'cost': [150], 'max_price_with_demand': [350], 'demand_den': [calculate_demand_den(150, 350)]},
  {'products': ['Basmati Rice (Large Sacks)'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Quinoa (Bulk Bags)'], 'cost': [60], 'max_price_with_demand': [130], 'demand_den': [calculate_demand_den(60, 130)]},
  {'products': ['Almond Milk (Cases)'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]},
  {'products': ['Greek Yogurts (Cases)'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Free-Range Eggs (Flats)'], 'cost': [50], 'max_price_with_demand': [110], 'demand_den': [calculate_demand_den(50, 110)]},
  {'products': ['Protein Bars (Boxes)'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]},
  {'products': ['Sparkling Water (Cases)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]},

  # Beauty & Personal Care (20)
  {'products': ['Salon Shampoos'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Deep Conditioners'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]}, # Adjusted cost
  {'products': ['Artisan Soaps'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Exfoliating Body Washes'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Anti-Aging Lotions'], 'cost': [55], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(55, 150)]},
  {'products': ['Mineral Sunscreens'], 'cost': [50], 'max_price_with_demand': [85], 'demand_den': [calculate_demand_den(50, 85)]}, # Adjusted cost
  {'products': ['Luxury Face Creams'], 'cost': [70], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(70, 300)]},
  {'products': ['Professional Makeup Kits'], 'cost': [100], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(100, 400)]},
  {'products': ['Designer Lipsticks'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Volumizing Mascaras'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Niche Perfumes'], 'cost': [80], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(80, 250)]},
  {'products': ['Premium Colognes'], 'cost': [75], 'max_price_with_demand': [220], 'demand_den': [calculate_demand_den(75, 220)]},
  {'products': ['Electric Toothbrushes'], 'cost': [50], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(50, 180)]},
  {'products': ['Water Flossers'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Whitening Mouthwashes'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Natural Deodorants'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Professional Hair Dryers'], 'cost': [70], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(70, 200)]},
  {'products': ['Ceramic Curling Irons'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Electric Razors'], 'cost': [60], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(60, 180)]},
  {'products': ['Luxury Shaving Creams'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost

  # Sports & Outdoors (20)
  {'products': ['Mountain Bikes'], 'cost': [300], 'max_price_with_demand': [1200], 'demand_den': [calculate_demand_den(300, 1200)]},
  {'products': ['Skateboards'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},
  {'products': ['Electric Scooters'], 'cost': [250], 'max_price_with_demand': [700], 'demand_den': [calculate_demand_den(250, 700)]},
  {'products': ['Camping Tents (4-person)'], 'cost': [100], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(100, 300)]},
  {'products': ['Down Sleeping Bags'], 'cost': [120], 'max_price_with_demand': [350], 'demand_den': [calculate_demand_den(120, 350)]},
  {'products': ['Internal Frame Backpacks'], 'cost': [150], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(150, 400)]},
  {'products': ['Waterproof Hiking Boots'], 'cost': [80], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(80, 250)]},
  {'products': ['Trail Running Shoes'], 'cost': [70], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(70, 180)]},
  {'products': ['Premium Yoga Mats'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Adjustable Dumbbells (Sets)'], 'cost': [200], 'max_price_with_demand': [600], 'demand_den': [calculate_demand_den(200, 600)]},
  {'products': ['Resistance Bands (Sets)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Official Size Basketballs'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['FIFA Quality Soccer Balls'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Graphite Tennis Rackets'], 'cost': [80], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(80, 250)]},
  {'products': ['Spinning Fishing Rods'], 'cost': [60], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(60, 180)]},
  {'products': ['Hard-Sided Coolers'], 'cost': [100], 'max_price_with_demand': [350], 'demand_den': [calculate_demand_den(100, 350)]},
  {'products': ['Insulated Water Bottles'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Folding Camping Chairs'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Adjusted cost
  {'products': ['LED Headlamps'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Compact Binoculars'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},

  # Toys & Games (20)
  {'products': ['Collectible Action Figures'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Fashion Dolls'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Large Building Block Sets'], 'cost': [60], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(60, 150)]},
  {'products': ['Strategy Board Games'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Trading Card Game Booster Boxes'], 'cost': [70], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(70, 150)]},
  {'products': ['1000-Piece Jigsaw Puzzles'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Die-Cast Model Cars'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Plush Stuffed Animals'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Kids Art Easels'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Wooden Play Kitchens'], 'cost': [80], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(80, 200)]},
  {'products': ['Electronic Keyboards (Toy)'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]},
  {'products': ['Hobby-Grade Remote Control Cars'], 'cost': [100], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(100, 300)]},
  {'products': ['New Release Video Games'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]},
  {'products': ['STEM Building Kits'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Outdoor Playhouses'], 'cost': [200], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(200, 500)]},
  {'products': ['Large Trampolines'], 'cost': [250], 'max_price_with_demand': [600], 'demand_den': [calculate_demand_den(250, 600)]},
  {'products': ['Backyard Swing Sets'], 'cost': [300], 'max_price_with_demand': [800], 'demand_den': [calculate_demand_den(300, 800)]},
  {'products': ['Inflatable Pool Floats'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]}, # Adjusted cost
  {'products': ['Large Kites'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Laser Tag Sets'], 'cost': [60], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(60, 150)]},

  # Books & Media (20)
  {'products': ['Hardcover Fiction Bestsellers'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Coffee Table Books'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Specialty Cookbooks'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Childrens Picture Books (Hardcover)'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Graphic Novels (Collected Editions)'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Magazine Subscriptions (Annual)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]},
  {'products': ['Vinyl Records (New Releases)'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['CD Box Sets'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Blu-ray Movie Collections'], 'cost': [60], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(60, 150)]},
  {'products': ['4K Ultra HD Discs'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Ebook Bundles'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Representing value
  {'products': ['Audiobook Credits (Monthly)'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Representing value
  {'products': ['Streaming Service Gift Cards'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]}, # Representing value
  {'products': ['Wall Calendars'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Leather-Bound Planners'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Hardcover Journals'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Fountain Pens'], 'cost': [50], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(50, 150)]},
  {'products': ['Framed Art Prints'], 'cost': [60], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(60, 200)]},
  {'products': ['Movie Posters (Licensed)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Board Game Expansions'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost

  # Automotive (20)
  {'products': ['Performance Car Tires (Set of 4)'], 'cost': [400], 'max_price_with_demand': [1000], 'demand_den': [calculate_demand_den(400, 1000)]},
  {'products': ['AGM Car Batteries'], 'cost': [150], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(150, 300)]},
  {'products': ['Beam Style Windshield Wipers (Pairs)'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Synthetic Motor Oils (5-Quart Jugs)'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Car Air Freshener Packs'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost (bulk)
  {'products': ['Custom Fit Car Covers'], 'cost': [100], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(100, 300)]},
  {'products': ['All-Weather Floor Mats (Sets)'], 'cost': [70], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(70, 200)]},
  {'products': ['Leatherette Seat Covers (Sets)'], 'cost': [120], 'max_price_with_demand': [300], 'demand_den': [calculate_demand_den(120, 300)]},
  {'products': ['Magnetic Phone Mounts'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Portable GPS Navigation Units'], 'cost': [80], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(80, 200)]},
  {'products': ['Dual Lens Dash Cams'], 'cost': [100], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(100, 250)]},
  {'products': ['Heavy Duty Jumper Cables'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Mechanics Tool Kits'], 'cost': [150], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(150, 500)]},
  {'products': ['Hydraulic Floor Jacks'], 'cost': [100], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(100, 250)]},
  {'products': ['Portable Tire Inflators'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Car Wash & Wax Kits'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]}, # Adjusted cost
  {'products': ['Ceramic Coating Sprays'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Orbital Polishers'], 'cost': [80], 'max_price_with_demand': [200], 'demand_den': [calculate_demand_den(80, 200)]},
  {'products': ['LED Headlight Conversion Kits'], 'cost': [60], 'max_price_with_demand': [150], 'demand_den': [calculate_demand_den(60, 150)]},
  {'products': ['Custom Taillight Assemblies'], 'cost': [150], 'max_price_with_demand': [400], 'demand_den': [calculate_demand_den(150, 400)]},

  # Office Supplies (20)
  {'products': ['Gel Pens (Dozen Packs)'], 'cost': [50], 'max_price_with_demand': [65], 'demand_den': [calculate_demand_den(50, 65)]}, # Adjusted cost
  {'products': ['Mechanical Pencils (Sets)'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Permanent Markers (Assorted Packs)'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Highlighters (Multi-Color Packs)'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Spiral Notebooks (5-Packs)'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Heavy Duty Binders'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['File Folders (Boxes of 100)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Electric Staplers'], 'cost': [50], 'max_price_with_demand': [90], 'demand_den': [calculate_demand_den(50, 90)]},
  {'products': ['Paper Clips (Large Tubs)'], 'cost': [50], 'max_price_with_demand': [60], 'demand_den': [calculate_demand_den(50, 60)]}, # Adjusted cost
  {'products': ['Sticky Notes (Bulk Packs)'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Mailing Envelopes (Boxes)'], 'cost': [50], 'max_price_with_demand': [75], 'demand_den': [calculate_demand_den(50, 75)]}, # Adjusted cost
  {'products': ['Printer Paper (Reams/Cases)'], 'cost': [50], 'max_price_with_demand': [80], 'demand_den': [calculate_demand_den(50, 80)]}, # Adjusted cost
  {'products': ['Mesh Desk Organizers'], 'cost': [50], 'max_price_with_demand': [70], 'demand_den': [calculate_demand_den(50, 70)]}, # Adjusted cost
  {'products': ['Ergonomic Office Chairs'], 'cost': [150], 'max_price_with_demand': [500], 'demand_den': [calculate_demand_den(150, 500)]},
  {'products': ['LED Desk Lamps'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Scientific Calculators'], 'cost': [50], 'max_price_with_demand': [120], 'demand_den': [calculate_demand_den(50, 120)]},
  {'products': ['Large Whiteboards'], 'cost': [80], 'max_price_with_demand': [250], 'demand_den': [calculate_demand_den(80, 250)]},
  {'products': ['Cork Bulletin Boards'], 'cost': [50], 'max_price_with_demand': [100], 'demand_den': [calculate_demand_den(50, 100)]},
  {'products': ['Cross-Cut Paper Shredders'], 'cost': [60], 'max_price_with_demand': [180], 'demand_den': [calculate_demand_den(60, 180)]},
  {'products': ['Locking File Cabinets'], 'cost': [120], 'max_price_with_demand': [350], 'demand_den': [calculate_demand_den(120, 350)]},
]


############ HUMAN GENERATED CODE ############

from pprint import pprint
from pickle import dump

products_list = [{k : v[0] for k, v in p.items()} for p in products_list]

def prettier_den(d):
  if d > 0.9:
    return round(d)
  if d > 0.09:
    return round(d, 1)
  if d < 0.01:
    return 0.01
  return round(d, 2)
for p in products_list: p['demand_den'] = prettier_den(p['demand_den'])

products = { k : [p[k] for p in products_list] for k in products_list[0] }

pprint(products_list)
pprint(products)

with open('BertrandCompetition.pkl', 'wb') as f:
    dump(products, f)
