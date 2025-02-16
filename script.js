// The slider javascript
let leftSide = document.querySelector('#left-side');
let masker = document.querySelector('.masker');
const handleMove = e => {
    let clientWidth = document.documentElement.clientWidth;
    let percent = (e.clientX / clientWidth) * 100;
    leftSide.style.width = `${percent}%`;
}

document.querySelector('.sliderContainer').onmousemove = handleMove;
// End of Slider

// Ai Code along with Canva

let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let isDrawing = false;
let model;

async function loadModel() {
  model = await tf.loadLayersModel("model/trained-model.json");
  console.log("Model loaded successfully");
  model.summary();
}
loadModel();

// Line Drawing Settings
ctx.strokeStyle = "black";
ctx.lineWidth = 14.0;
ctx.lineCap = "round";
let drawing = false;
// Drawing code
const getMouse = (e) => {
  const canvaPos = canvas.getBoundingClientRect();
  return {
    x: e.clientX - canvaPos.left,
    y: e.clientY - canvaPos.top,
  };
};

const handleClick = (e) => {
  drawing = true;
  ctx.beginPath();
  let pos = getMouse(e);
  ctx.moveTo(pos.x, pos.y);
};
const handleDraw = (e) => {
  if (!drawing) return;
  let pos = getMouse(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
};
// Handle the drawing movements
canvas.onmousedown = handleClick;
canvas.onmousemove = handleDraw;
canvas.onmouseup = () => (drawing = false);
canvas.onmouseleave = () => drawing = false;

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("prediction").innerText = "Prediction: ?";
}

async function predict() {
  let imageData = ctx.getImageData(0, 0, 280, 280);

  let tensor = tf.browser
    .fromPixels(imageData, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .sub(tf.scalar(1.0))
    .mul(tf.scalar(-1.0)) // Invert colors
    .expandDims(0);

  let predictions = model.predict(tensor);

  let predictedDigit = predictions.argMax(1).dataSync()[0];
  document.getElementById("prediction").innerText =
    "Prediction: " + predictedDigit;
}
