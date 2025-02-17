// Force TensorFlow.js to use WebGL for GPU acceleration
(async () => {
  await tf.setBackend('webgl');
  console.log("Using TensorFlow.js backend:", tf.getBackend());
})();

console.log("Hello TensorFlow");
import { MnistData } from "./data.js";

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("train-button").addEventListener("click", run);
});

async function showExamples(data) {
  const surface = tfvis.visor().surface({ name: "Input Data Examples", tab: "Input Data" });
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs.slice([i, 0], [1, examples.xs.shape[1]]).reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);
    imageTensor.dispose();
  }
}

async function run() {
  await tf.setBackend('webgl');
  
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  const model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture", tab: "Model" }, model);

  const batchSize = parseInt(document.getElementById("batch-size").value) || 64;
  const epochs = parseInt(document.getElementById("epochs").value) || 10;

  await train(model, data, batchSize, epochs);
  await saveModel(model);
  await showAccuracy(model, data);
  await showConfusion(model, data);
}

function getModel() {
  const model = tf.sequential();
  const IMAGE_WIDTH = 28, IMAGE_HEIGHT = 28, IMAGE_CHANNELS = 1;

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: "relu",
    kernelInitializer: "varianceScaling",
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 32,
    strides: 1,
    activation: "relu",
    kernelInitializer: "varianceScaling",
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({ optimizer: tf.train.adam(), loss: "categoricalCrossentropy", metrics: ["accuracy"] });

  return model;
}

async function train(model, data, batchSize, epochs) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
      name: "Model Training",
      tab: "Model",
      styles: { height: "1000px" },
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const TRAIN_DATA_SIZE = 55000;
  const TEST_DATA_SIZE = 10000;

  const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  let progressDiv = document.getElementById("progress");
  if (!progressDiv) {
      progressDiv = document.createElement("div");
      progressDiv.id = "progress";
      document.body.appendChild(progressDiv);
  }

  console.log("Training started...");

  await model.fit(trainXs, trainYs, {
      batchSize: batchSize,
      validationData: [testXs, testYs],
      epochs: epochs,
      shuffle: true,
      callbacks: {
          onEpochEnd: async (epoch, logs) => {
              let message = `Epoch ${epoch + 1}/${epochs} - Accuracy: ${logs.acc.toFixed(4)} - Loss: ${logs.loss.toFixed(4)}`;
              if (logs.val_acc !== undefined) {
                  message += ` - Val Accuracy: ${logs.val_acc.toFixed(4)}`;
              }
              if (logs.val_loss !== undefined) {
                  message += ` - Val Loss: ${logs.val_loss.toFixed(4)}`;
              }

              console.log(message);
              progressDiv.innerText = message;
          }
      }
  });

  console.log("Training completed.");
  progressDiv.innerText = "Training Completed!";
}

async function saveModel(model) {
  await model.save('downloads://mnist_model');
  console.log("Model saved!");
}

const classNames = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  tfvis.show.perClassAccuracy({ name: "Accuracy", tab: "Evaluation" }, classAccuracy, classNames);
  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  tfvis.render.confusionMatrix({ name: "Confusion Matrix", tab: "Evaluation" }, { values: confusionMatrix, tickLabels: classNames });
  labels.dispose();
}

function doPrediction(model, data, testDataSize = 500) {
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, 28, 28, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);
  testxs.dispose();
  return [preds, labels];
}
