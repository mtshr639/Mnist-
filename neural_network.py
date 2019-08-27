import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline

#NNクラスの定義
class neuralNetwork:
    
    
    #初期化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # 重み 入力層ー隠れ層間 wih 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        # 重み 隠れ層ー出力層間
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # 学習率
        self.lr = learningrate
        # 活性化関数はシグモイド関数を使用
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # NNの学習
    def train(self, inputs_list, targets_list):
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 隠れ層への入力を計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隠れ層からの出力を計算
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 出力層への入力を計算
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 出力層からの最終的な出力を計算
        final_outputs = self.activation_function(final_inputs)
        
        # 出力層の誤差 = 目標 - 出力層からの出力
        output_errors = targets - final_outputs
        # 隠れ層の誤差は出力層の誤差を重みの割合で分配
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # 隠れ層ー出力層間の重みの更新
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # 入力層ー隠れ層間の重みの更新
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # NNへの照会
    def query(self, inputs_list):
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # 隠れ層への入力を計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隠れ層からの出力を計算
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 出力層への入力を計算
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 出力層からの出力を計算
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

#入力画像サイズは28*28＝784
#隠れ層サイズは可変 とりあえず100
#出力層は0~9の確率を表す サイズは10
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
lerning_rate = 0.1
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, lerning_rate)

#Mnist 訓練データセットを読み込んでリストにする
training_data_file = open("mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#NNの学習
#トレーニングデータ全体に対して実行

for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

#Mnist テストデータセットを読み込んでリストにする
test_data_file = open("mnist_dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#テストデータセットの最初のデータを取得
all_values = test_data_list[0].split(',')
print(all_values[0])

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

#ニューラルネットワークのテスト

#Scoredは判定リスト 
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    #正解は配列の1番目
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    # 入力値のスケーリングシフト
    inputs =(numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    
    #最大値のインデックスをラベルに対応させる
    label = numpy.argmax(outputs)
    print(label, "netwotk's answer")
    
    #正解(1) 不正解(0) をリストに追加
    if (label == correct_label):
        #正解なら1を追加
        scorecard.append(1)
    else:
        #間違いなら0を追加
        scorecard.append(0)
        pass
    pass

#最後に判定成功率を出力
scorecard_array = numpy.asarray(scorecard)
print("performance = ",scorecard_array.sum() / scorecard_array.size)