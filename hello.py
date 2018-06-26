from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import cf_deployment_tracker
import os
import json

# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

db_name = 'mydb'
client = None
db = None

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

@app.route('/')
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if(request.method == 'POST'):
        file = request.files['image'].read()

        if not file:
            return render_template('index.html', label="No file")
   
        image = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_UNCHANGED)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(image, 10, 250)
        (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0

        def sort_contours(cnts, method="left-to-right"):
            # initialize the reverse flag and sort index
            reverse = False
            i = 0
        
            # handle if we need to sort in reverse
            if method == "right-to-left" or method == "bottom-to-top":
                reverse = True
        
            # handle if we are sorting against the y-coordinate rather than
            # the x-coordinate of the bounding box
            if method == "top-to-bottom" or method == "bottom-to-top":
                i = 1
        
            # construct the list of bounding boxes and sort them from top to
            # bottom
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                key=lambda b:b[1][i], reverse=reverse))
        
            # return the list of sorted contours and bounding boxes
            return cnts

        cnts = sort_contours(cnts)

        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w>50 and h>50:
                idx+=1
                new_img=image[y:y+h,x:x+w]
                cv2.imwrite(str(idx) + '.jpg', new_img)
        #cv2.imshow("im",image)
        cv2.waitKey(0)
        features = []
        for i in range(1, idx+1):
                image = cv2.imread(str(i)+'.jpg', 0)
                resized = cv2.resize(image, (45, 45), interpolation = cv2.INTER_AREA)
                feature = np.array(resized).flatten()
                features.append(feature)
        features = pd.DataFrame(features)
        predictions = model.predict(features)
        myListInitial = predictions.tolist()
        myList = []
        for var in myListInitial:
            if var == '11':
                myList.append('+')
            elif var == '12':
                myList.append('-')
            elif var == '13':
                myList.append('*')
            elif var == '14':
                myList.append('/')
            else:
                myList.append(var)
        str1 = ''.join(myList)
        ans = eval(str1)     
        return render_template('index.html', label=ans)




@atexit.register
def shutdown():
    if client:
        client.disconnect()

if __name__ == '__main__':
    model = joblib.load('modelKaggle1.pkl')
    app.run(host='0.0.0.0', port=port, debug=True)
