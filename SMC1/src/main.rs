use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

enum Model {
    L1,
    L2,
    Poisson,
}

struct Regression {
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    alpha: f64,
    model: Model,
    weight: Array1<f64>,
}

impl Regression {
    fn new(epochs: usize, learning_rate: f64, batch_size: usize, alpha: f64, model: Model) -> Self {
        let weight = Array1::from_elem(1, 0.0);
        Regression { epochs, learning_rate, batch_size, alpha, model, weight }
    }

    fn train(&mut self, x_train: Array2<f64>, y_train: Array1<f64>, x_test: Array2<f64>, y_test: Array1<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let dim = x_train.shape()[1];
        self.weight = Array1::from_shape_vec((dim,), vec![0.001; dim]).unwrap();
        let mut train_losses = vec![];
        let mut test_losses = vec![];
        let mut train_mse = vec![];
        let mut test_mse = vec![];
        let mut rng = thread_rng();
        for epoch in 0..self.epochs {
            let train_loss = self.sgd(&mut rng, &x_train, &y_train);
            let (test_loss, _) = match self.model {
                Model::L1 => self.compute_loss_l1(&x_test, &y_test),
                Model::L2 => self.compute_loss_l2(&x_test, &y_test),
                Model::Poisson => self.poisson_regression(&x_test, &y_test),
            };
            train_losses.push(train_loss);
            test_losses.push(test_loss);
            train_mse.push(mse(&x_train.dot(&self.weight), &y_train));
            test_mse.push(mse(&x_test.dot(&self.weight), &y_test));
            println!("{:d}\t->\tTrainL : {:.7}\t|\tTestL : {:.7}\t|\tTrainMSE : {:.7}\t|\tTestMSE: {:.7}", epoch, train_loss, test_loss, train_mse[epoch], test_mse[epoch]);
        }
        (train_losses, test_losses, train_mse, test_mse)
    }

    fn sgd(&mut self, rng: &mut impl rand::Rng, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let mut losses = vec![];
        let mut indices = (0..x.shape()[0]).collect::<Vec<_>>();
        indices.shuffle(rng);
        for i in (0..x.shape()[0]).step_by(self.batch_size) {
            let batch_indices = indices[i..i+self.batch_size].to_vec();
            let x_batch = x.select(ndarray::Axis(0), &batch_indices);
            let y_batch = y.select(ndarray::Axis(0), &batch_indices);
            let (loss, dw) = self.poisson_regression(&x_batch, &y_batch);
            self.weight -= self.learning_rate * &dw;
            losses.push(loss);
        }
        losses.iter().sum::<f64>() / losses.len() as f64
    }
    fn compute_loss_l1(&self, x: &Array2<f32>, y: &Array1<f32>) -> (f32, Array1<f32>) {
        let samples = x.shape()[0];
        let y_star = x.dot(&ArrayView1::from(&self.weight)).to_owned();
        let loss = (y_star - y).mapv(|x| x.powi(2)).sum();
        let reg_loss = self.alpha * self.weight.iter().map(|&x| x.abs()).sum();
        let total_loss = loss + reg_loss;

        let grad = (-2.0) * x.t().dot(&(y - &y_star)) + self.alpha;

        (total_loss / (samples as f32), grad)
    }

    fn compute_loss_l2(&self, x: &Array2<f32>, y: &Array1<f32>) -> (f32, Array1<f32>) {
        let samples = x.shape()[0];
        let y_star = x.dot(&ArrayView1::from(&self.weight)).to_owned();
        let loss = (y_star - y).mapv(|x| x.powi(2)).sum();
        let reg_loss = self.alpha * self.weight.iter().map(|&x| x.powi(2)).sum();
        let total_loss = loss + reg_loss;

        let grad = 2.0 * x.t().dot(&(y_star - y)) + 2.0 * self.alpha * self.weight.l1_norm();

        (total_loss / (samples as f32), grad)
    }

    fn poisson_regression(&self, x: &Array2<f32>, y: &Array1<f32>) -> (f32, Array1<f32>) {
        let samples = x.shape()[0];
        let y_star = x.dot(&ArrayView1::from(&self.weight)).to_owned();
        let loss = x.dot(&y_star.mapv(f32::exp)) - y.dot(&y_star);
        let total_loss = loss.sum();

        let grad = x.t().dot(&(y_star.mapv(f32::exp)) - &x.t().dot(&y));

        (total_loss / (samples as f32), grad)
    }
}
fn load_music_data(fname: &str, add_bias: bool) -> (Array2<i32>, Array2<f32>, Array2<i32>, Array2<f32>) {
    let data: Array2<f32> = ndarray::ArrayBase::from_file(fname, ",", |s| s.parse::<f32>().unwrap()).unwrap();

    let x1 = data.slice(s![..800, 1..]).to_owned();
    let y1 = data.slice(s![..800, 0]).map(|v| v as i32).to_owned();
    let x2 = data.slice(s![800.., 1..]).to_owned();
    let y2 = data.slice(s![800.., 0]).map(|v| v as i32).to_owned();

    if add_bias {
        let y_mean = y1.mean().unwrap();
        let y1 = y1 - (y_mean as i32);
        let y2 = y2 - (y_mean as i32);
        (y1, x1, y2, x2)
    } else {
        (y1, x1, y2, x2)
    }
}

fn music_mse(pred: ArrayView1<i32>, gt: ArrayView1<i32>) -> f32 {
    let diff = gt.into_shape((gt.len(),)).unwrap() - &pred;
    let mse = diff.mapv(|x| x * x).mean().unwrap();
    mse
}

fn analyse_data(train_y: ArrayView1<i32>, test_y: ArrayView1<i32>) {
    let mut data = Array::zeros(train_y.len() + test_y.len());
    data.slice_mut(s![..train_y.len()]).assign(&train_y);
    data.slice_mut(s![train_y.len()..]).assign(&test_y);

    let hist = plt::histogram(&data, 90);
    let bins = hist.bins;
    let weights = hist.weights;

    plt::plot(&bins, &weights, "b-");
    plt::title("Histogram of the labels in the train and test set");
    plt::xlabel("Label of Year");
    plt::ylabel("Number");
    plt::show().unwrap();
}

fn create_fig(train_losses: &[f32], test_losses: &[f32], train_mse: &[f32], test_mse: &[f32]) {
    let fig = plt::subplots(1, 2);

    let ax1 = fig.axes.get_mut(0).unwrap();
    ax1.plot(train_losses, Some("b-"), Some("Train loss"));
    ax1.plot(test_losses, Some("r-"), Some("Test loss"));
    ax1.legend();
    ax1.set_title("Average Train Cross Entropy Loss");
    ax1.set_xlabel("Number of Epochs");
    ax1.set_ylabel("Cross Entropy Loss");

    let ax2 = fig.axes.get_mut(1).unwrap();
    ax2.plot(train_mse, Some("b-"), Some("Train MSE"));
    ax2.plot(test_mse, Some("r-"), Some("Test MSE"));
    ax2.legend();
    ax2.set_title("Mean MSE varying with Epochs");
    ax2.set_xlabel("Number of Epochs");
    ax2.set_ylabel("Mean MSE");

    fig.show().unwrap();
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Main method

    let fname = "/Users/wdaugherty/Cornell_Tech_DL/CS-5787/HW2/data/YearPredictionMSD.npy";
    let data: ndarray::Array2<f64> = ndarray_npy::from_path(fname)?;
    let train_years = &data.slice(s![..800, 0]).to_owned();
    let train_feat = &data.slice(s![..800, 1..]).to_owned();
    let test_years = &data.slice(s![800.., 0]).to_owned();
    let test_feat = &data.slice(s![800.., 1..]).to_owned();

    let year_mean = train_years.mean_axis(Axis(0)).unwrap();
    let year_std = train_years.std_axis(Axis(0), 0.0);
    let train_years = train_years
        .mapv(|x| (x - 1992.0) / year_std)
        .to_owned();
    let test_years = test_years.mapv(|x| (x - 1992.0) / year_std).to_owned();
    println!("train_years: {}", train_years);
    println!("test_years: {}", test_years);

    let all_feat = ndarray::stack(Axis(0), &[train_feat, test_feat]).unwrap();
    let feat_mean = all_feat.mean_axis(Axis(0)).unwrap();
    let feat_std = all_feat.std_axis(Axis(0), 0.0);
    let train_feat = train_feat
        .mapv(|x| (x - &feat_mean) / &feat_std)
        .to_owned();
    let test_feat = test_feat
        .mapv(|x| (x - &feat_mean) / &feat_std)
        .to_owned();

    println!("---{}----", train_years.len());

    let epochs = 50;
    let learning_rate = 0.000001;
    let batch_size = 10;
    let alpha = 0.01;
    let model = "L1";
    let momentum = 0.005;

    // Problem 3-(2~4)
    let mut rg = Regression::new(
        epochs,
        learning_rate,
        batch_size,
        alpha,
        model,
        momentum,
        train_feat.shape()[1],
    );
    let (train_losses, test_losses, train_mse, test_mse) =
        rg.train(&train_feat, &train_years, &test_feat, &test_years);
    create_fig(train_losses, test_losses, train_mse, test_mse)?;

    Ok(())
}

