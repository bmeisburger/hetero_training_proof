from models.list_models import *

class Model:
  def get(params):
    model = None
    if params["name"] == "ROASTMLP":
      model = ROASTFCN(params["ROASTMLP"]["input_dim"], 
                          params["ROASTMLP"]["num_layers"], 
                          params["ROASTMLP"]["hidden_size"], 
                          params["ROASTMLP"]["num_class"], 
                          params["ROASTMLP"]["compression"], 
                          params["ROASTMLP"]["seed"])
    elif params["name"] == "ROASTCNN":
      model = ROASTCNN(in_channels=params["ROASTCNN"]["in_channels"], 
                          out_channels=params["ROASTCNN"]["out_channels"], 
                          num_layers=params["ROASTCNN"]["num_layers"], 
                          hidden_size=params["ROASTCNN"]["hidden_size"], 
                          num_class=params["ROASTCNN"]["num_class"], 
                          kernel_size=params["ROASTCNN"]["kernel_size"], 
                          compression=params["ROASTCNN"]["compression"], 
                          seed=params["ROASTCNN"]["seed"])
    else:
      raise NotImplementedError
    return model


