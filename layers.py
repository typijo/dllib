import numpy as np
import logging

class Layer:
    ID = 0
    def __init__(self, name=None, **kwargs):
        self._parents = []
        self._children = []
        self._dim = 0
        
        if name is None:
            self._name = "%d_%s" % (Layer.ID, self.__class__.__name__)
            Layer.ID += 1
        else:
            self._name

        self._in = None
        self._val = None
        self._grad = None
        
    @property
    def dim(self):
        return self._dim
    
    @property
    def val(self):
        return self._val
    
    @property
    def grad(self):
        if self._grad is None:
            return np.ones(self._dim)
        else:
            return self._grad
        
    def build(self, child, **kwargs):
        if child:
            self._children.append(child)
        
        for p in self._parents:
            p.build(self, **kwargs)
    
    def fw(self, **kwargs):
        pass
    
    def bp(self, **kwargs):
        pass

    def reset_grad(self, **kwargs):
        self._grad = None
        for p in self._parents:
            p.reset_grad()
    
    def __str__(self):
        return self._name
    
    def summary(self):
        print(str(self) + ": shaped " + str(self._dim) + ", parent " + " / ".join([str(p) for p in self._parents]))
        for p in self._parents:
            p.summary()
        

class Input(Layer):
    def __init__(self, dim=32, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
    
    def fw(self, val, **kwargs):
        val = np.array(val)
        if val.shape[0] != self._dim:
            raise ValueError(
                "input shape not matched: {} and {}".format(val.shape[0], self._dim))
        
        self._val = val
        for c in self._children:
            c.fw()

class Output(Layer):
    def __init__(self, tensor_in, **kwargs):
        super().__init__(**kwargs)
        self._parents = [tensor_in]
        self._dim = tensor_in.dim
    
    def bp(self, opt, **kwargs):
        for p in self._parents:
            grad = np.ones(self._dim)
            p.bp(opt=opt, grad=grad, **kwargs)
    
    @property
    def val(self):
        return self._parents[0].val


class Dense(Layer):
    def __init__(self, tensor_in, dim_out=64, init_W=None, init_b=None, **kwargs):
        super().__init__(**kwargs)
        self._parents = [tensor_in]
        
        if init_W is not None:
            self._W = np.array(init_W)
            if self._W.shape != (dim_out, tensor_in.dim):
                raise ValueError("init_w has different shape with %s" % str((dim_out, tensor_in.dim)))
        else:
            self._W = np.random.normal(size=(dim_out, tensor_in.dim))
            
        if init_b is not None:
            self._b = np.array(init_b)
        else:
            self._b = np.array(init_b) or np.random.normal(size=(dim_out,))
        
        self._dim = dim_out
    
    def fw(self, **kwargs):
        if self._parents[0].val is None:
            logging.info("parent variable 0 is None, skip")
            return
        
        self._in = self._parents[0].val
        self._val = self._W @ self._in.T + self._b.T
        
        for c in self._children:
            c.fw()
    
    def bp(self, opt, grad, **kwargs):
        dw = self._in.reshape((1, -1))
        db = np.ones(self._b.shape)
        dx = self._W

        self._grad = {
            "W": np.dot(grad.reshape(-1, 1), dw),
            "b": grad * db,
            "x": np.dot(dx.T, grad)
        }

        self._W = opt(self._W, self._grad["W"])
        self._b = opt(self._b, self._grad["b"])

        for p in self._parents:
            p.bp(opt=opt, grad=self._grad["x"])
    
    @property
    def grad(self):
        if self._grad is None:
            return None
        return self._grad["x"]


class Sigmoid(Layer):
    def __init__(self, tensor_in, **kwargs):
        super().__init__(**kwargs)
        self._parents = [tensor_in]
        self._dim = tensor_in.dim

    @classmethod
    def _sig(cls, val):
        return 1 / (1 + np.exp(-val))
    
    def fw(self, **kwargs):
        if self._parents[0].val is None:
            logging.info("parent variable 0 is None, skip")
            return
        
        self._in = self._parents[0].val
        self._val = Sigmoid._sig(self._in)
    
        for c in self._children:
            c.fw()
    
    def bp(self, opt, grad, **kwargs):
        self._grad = Sigmoid._sig(self._in) * (1 - Sigmoid._sig(self._in))

        for p in self._parents:
            p.bp(opt=opt, grad=grad * self._grad)


class MeanSquaredError(Layer):
    def __init__(self, tensor_pred, tensor_ans, **kwargs):
        super().__init__(**kwargs)
        self._parents = [tensor_pred, tensor_ans]

        if tensor_pred.dim != tensor_ans.dim:
            raise ValueError("two tensors should be same size")

        self._dim = tensor_pred.dim
    
    def fw(self, **kwargs):
        if self._parents[0].val is None:
            logging.debug("parent variable 0 is None, skip")
            return
        if self._parents[1].val is None:
            logging.debug("parent variable 0 is None, skip")
            return
        
        val_pred = self._parents[0].val
        val_ans = self._parents[1].val

        self._val = (1 / 2) * np.sum((val_ans - val_pred) ** 2)

        for c in self._children:
            c.fw()

    def bp(self, opt, grad, **kwargs):
        val_pred = self._parents[0].val
        val_ans = self._parents[1].val

        self._grad = -(val_ans - val_pred)

        for p in self._parents:
            p.bp(opt=opt, grad=self._grad * grad)


class Model:
    def __init__(self, layer_in, layer_pred, layer_ans, lossfunc, opt):
        self._layers = {}
        self._opt = opt
        
        self._lin = layer_in
        self._lans = layer_ans
        self._lpred = layer_pred
        self._loss = lossfunc(self._lpred, self._lans)
        self._lout = Output(self._loss)
        
        q = [self._lout]
        while len(q) > 0:
            l_this = q.pop()

            name = str(l_this)
            if name in self._layers:
                raise NameError("Layer name %s duplicates" % name)
            self._layers[name] = l_this

            for p in l_this._parents:
                q.append(p)
        
        self._lout.build(None)
        self._lout.summary()
    
    def layer(self, name):
        return self._layers[name]
    
    def fit(self, Xs, ys, epochs=5):
        # just online training
        loss = []
        n = 0
        for i in range(epochs):
            for X, y in zip(Xs, ys):
                self._lin.fw(X)
                self._lans.fw(y)

                self._lout.bp(self._opt)
                self._lout.reset_grad()

                loss += [self._loss.val]
                n += 1
                if n % 100 == 0:
                    logging.info("avg loss in 100 iters: %f" % (sum(loss) / n))
                    loss = []
                    n = 0
                
            logging.info("Epoch %d finished" % i)

    def predict(self, X):
        self._lin.fw(X)
        return self._lpred.val
