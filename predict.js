
let myChart;

$('#image-selector').change(function(){
    let reader = new FileReader();
    reader.onload = function(){
        let dataURL = reader.result;
        $('#selected-image').attr('src', dataURL);
    }

    let file = $('#image-selector').prop('files')[0];
    reader.readAsDataURL(file);
});

function createChart(top5){
    let data = new Array();
    let labels = new Array();
    top5.forEach(p => {
        labels.push(p.className);
        data.push((p.probability*100).toFixed(2));
    });

    var ctx = document.getElementById('myChart').getContext('2d');
    myChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ]
                
            }]
        }
    });

   

}

let model;
(async function(){
    console.log("verison log:", tf.version);
    console.log("tf methods: ", tf);
    console.log("Model loading...");
    document.getElementById('p_bar').style.display = "block";
    model = await tf.loadLayersModel('./tfjs-models/mobilenet/model.json');
    document.getElementById('p_bar').style.display = "none";
    console.log("model loaded");
})();

$('#predict-button').click(function(){
    prediction();
});

async function prediction(){
    let image = $('#selected-image').get(0);
   
    let tensor = preprocess_image(image);
    document.getElementById('spinner').style.display='block';
    let predictions = await model.predict(tensor).data();
    document.getElementById('spinner').style.display = 'none';

    let top5 = Array.from(predictions)
        .map(function(p, i){
            return{
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function(a, b){
            return b.probability - a.probability;
        }).slice(0, 5);

    console.log("Predictions: " , top5[1].probability);
    console.log("Classes: " , top5[1].className);
    
    clearChart();
    createChart(top5);
}

function clearChart(){
    
    if (typeof myChart != 'undefined') {
        myChart.destroy();
   }
}

function preprocess_image(image){
    let tensor = tf.browser.fromPixels(image);
    tensor = tf.image.resizeNearestNeighbor(tensor, [224, 224]).toFloat();

    let offset = tf.scalar(127.5);
    tensor = tensor.sub(offset);
    tensor = tensor.div(offset);

    tensor = tensor.expandDims();

    return tensor;
}
