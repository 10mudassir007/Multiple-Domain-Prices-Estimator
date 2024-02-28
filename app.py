from flask import Flask, render_template, request
from model_laptops import predict_laptops
from model_cars import predict_cars
from model_houses import predict_houses
from model_mobiles import predict_mobiles
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/cars', methods=['GET', 'POST'])
def cars():
    if request.method == 'POST':
        # Handle the form submission and prediction logic here
        # You can access form data using request.form
        doors = int(request.form.get('doornumber'))
        hp = int(request.form.get('horsepower'))
        rpm = int(request.form.get('peakrpm'))
        citympg = int(request.form.get('citympg'))
        highwaympg = int(request.form.get('highwaympg'))
        fueltype = request.form.get('fueltype')
        carbody = request.form.get('carbody')
        drivewheel = request.form.get('drivewheel')
        price_cars = predict_cars(doors,hp,rpm,citympg,highwaympg,fueltype,carbody,drivewheel)
        return render_template('cars.html',prediction_cars=price_cars)
    else:
        # Render the form for GET requests
        return render_template('cars.html')
    
@app.route('/laptops', methods=['GET', 'POST'])
def laptops():
    if request.method == 'POST':
        brandname = request.form.get('brand')
        core = int(request.form.get('core'))
        ram = int(request.form.get('ram'))
        storage = int(request.form.get('Storage'))
        screen = float(request.form.get('screen'))
        price_laptops = predict_laptops(brandname,storage,ram,screen,core)
        return render_template('laptops.html',prediction_laptops=price_laptops)
    else:
        return render_template('laptops.html')
    
@app.route('/mobiles', methods=['GET', 'POST'])
def mobiles():
    if request.method == 'POST':
        brandname = request.form.get('Brand')
        ram = int(request.form.get('ram'))
        scs = float(request.form.get('screenSize'))
        storage = int(request.form.get('Internal Storage'))
        resx = int(request.form.get('resloution x'))
        resy = int(request.form.get('resloution y'))
        battery = int(request.form.get('battery'))
        rec = int(request.form.get('rear camera'))
        frc = int(request.form.get('front camera'))
        cpu = int(request.form.get('cpu'))
        price_mobiles = predict_mobiles(brandname,battery,scs,resx,resy,cpu,ram,storage,rec,frc)
        
        return render_template('mobiles.html',prediction_mobiles=price_mobiles)
    else:
        return render_template('mobiles.html')

@app.route('/houses', methods=['GET', 'POST'])
def houses():
    if request.method == 'POST':
        size = int(request.form.get('house_size'))
        rooms = int(request.form.get('rooms'))
        brooms = int(request.form.get('brooms'))
        stories = int(request.form.get('stories'))
        parking = int(request.form.get('parking'))
        mroad = request.form.get('mainroad')
        groom = request.form.get('guestroom')
        basement = request.form.get('basement')
        hotw = request.form.get('hotwaterheating')
        ac = request.form.get('airconditioning')
        parea = request.form.get('prefarea')
        furnish = request.form.get('furnishingstatus')
        price_houses = predict_houses(size,rooms,brooms,stories,parking,mroad,groom,basement,hotw,ac,parea,furnish)
        return render_template('house.html',prediction_houses=price_houses)
    else:
        return render_template('house.html')


if __name__ == '__main__':
    app.run(debug=True,port=5500)
