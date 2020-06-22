import Utiles.MlFlow as mlFlow

d1 = {}
d1['z'] = 2

mlFlow1 = mlFlow.MlFlow('proj_1')

# mlFlow1.log_parameter('xyz', 17)
# mlFlow1.NewExperiment()
# mlFlow1.log_parameter('xyz', 23)
# mlFlow1.log_parameter('abc', 55)
# mlFlow1.log_parameter('abc', 45)
# mlFlow1.log_metric('mse', 87)

# mlFlow1.ToCsv()
# mlFlow1.log_files(['D:/ML/R&D/EventSeriesClassification/Project1/mlFlowTest.py'])
mlFlow1.log_files(['./mlFlowTest.py', './../SeriesFeatures.py'])
print('kuku')
# mlFlow1.
