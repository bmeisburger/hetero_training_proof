from models.list_models import *

class Model:
  def get(params):
    model = None
    if params["name"] == "ROASTMLP":
      model = ROASTFCN(params["ROASTMLP"]["input_dim"], params["ROASTMLP"]["num_layers"], params["ROASTMLP"]["hidden_size"], params["ROASTMLP"]["num_class"], params["ROASTMLP"]["compression"], params["ROASTMLP"]["seed"])
    else:
      raise NotImplementedError
    return model


