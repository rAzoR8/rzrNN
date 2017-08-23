#pragma once

#include <vector>
#include <stdint.h>
#include <functional>
#include <iostream>
#include <algorithm>

class Neuron
{
	friend class Layer;
	friend class Network;
public:
	Neuron(const std::vector<uint32_t>& _Inputs, const std::vector<float>& _Weights, const float _fBias = 0.f) :
		m_InputIndices(_Inputs), m_Weights(_Weights), m_fBias(_fBias) {}
	Neuron(const std::vector<uint32_t>& _Inputs, const float _fBias = 0.f) :
		m_InputIndices(_Inputs), m_Weights(m_InputIndices.size(), 0.5f), m_fBias(_fBias) {}

	// input interpreded as bias
	Neuron(const float _fInput = 0.f) : m_fBias(_fInput) {}
	~Neuron() {}

	void SetInput(const float _fInput) { m_fBias = _fInput; }

	std::vector<uint32_t>& GetInputs() { return m_InputIndices; }
	std::vector<float>& GetWeights() { return m_Weights; }

private:
	float m_fActivation = 0.f; // a_l = sigmoid(z_l)
	float m_fError = 0.f; // d_l
	float m_fBias = 0.f;
	float m_fWeightedOutput = 0.f; // z_l
	std::vector<uint32_t> m_InputIndices;
	std::vector<float> m_Weights;
};

using FloatFunc = std::function<float(const float)>;
using FloatFloatFunc = std::function<float(const float, const float)>;

FloatFunc Sigmoid = [](const float fZ)->float {return 1.f / (1.f + expf(-fZ)); };
FloatFunc SigmoidPrime = [](const float fZ)->float { float z = Sigmoid(fZ); return z * (1.f-z); };

FloatFunc Relu = [](const float fZ)->float {return fZ < 0.f ? 0.f : fZ; };
FloatFunc ReluPrime = [](const float fZ)->float {return fZ < 0.f ? 0.f : 1.f; };

FloatFloatFunc QuadraticCost = [](const float _Aj, const float _Yj) -> float { float dif = _Yj - _Aj; return 0.5f * (dif*dif); };
FloatFloatFunc QuadraticCostPrime = [](const float _Aj, const float _Yj) -> float { return _Aj -_Yj; };

FloatFunc g_Activation[] = { Sigmoid, Relu };
FloatFunc g_ActivationPrime[] = { SigmoidPrime, ReluPrime };

FloatFloatFunc g_Cost[] = { QuadraticCost };
FloatFloatFunc g_CostPrime[] = { QuadraticCostPrime };

enum EActivationFunction
{
	kSigmoid = 0,
	kRelu,
};

enum ECostFunction
{
	kMSE = 0,
};

class Layer
{
	friend class Network;
public:
	Layer(const uint32_t _uNeurons, const EActivationFunction _kActivation = kSigmoid) :
		m_Neurons(_uNeurons), m_kActivation(_kActivation){}
	~Layer() {}

	void FeedForward(const std::vector<float>& _X);
	// _X = input , _Y expected approximation
	void BackProp(
		const std::vector<float>& _X,
		const std::vector<float>& _Y,
		const float _fLearningRate,
		const uint32_t _uTrainingSetSize,
		const ECostFunction _kCost = kMSE); // takes truth _Y

	Layer* GetOutputLayer() { return m_pNextLayer == nullptr ? this : m_pNextLayer->GetOutputLayer(); }

	void SetNext(Layer* _pNext) { m_pNextLayer = _pNext; }
	void SetPrev(Layer* _pPrev) { m_pPrevLayer = _pPrev; }

	ptrdiff_t ArgMax();

protected:
	std::vector<Neuron> m_Neurons;
	EActivationFunction m_kActivation;
	Layer* m_pPrevLayer = nullptr;
	Layer* m_pNextLayer = nullptr;
};

inline void Layer::FeedForward(const std::vector<float>& _X)
{
	// set inputs
	const size_t inputSize(_X.size());

	if (inputSize != m_Neurons.size())
		return;

	for (size_t i = 0; i < inputSize; ++i)
	{
		m_Neurons[i].SetInput(_X[i]);
	}

	const FloatFunc& h(g_Activation[m_kActivation]);

	Layer* pCurrent = this;
	Layer* pPrev = m_pPrevLayer;

	do
	{	
		for (Neuron& n : pCurrent->m_Neurons)
		{
			n.m_fError = 0.f; // reset error
			n.m_fWeightedOutput = n.m_fBias; // also for input neurons
			if (pPrev != nullptr)
			{
				const size_t uInputCount(n.m_InputIndices.size());
				for (size_t i = 0; i < uInputCount; ++i)
				{
					float fInput = pPrev->m_Neurons[n.m_InputIndices[i]].m_fWeightedOutput;
					fInput *= n.m_Weights[i];
					n.m_fWeightedOutput += fInput;
				}
			}
			// activation
			n.m_fActivation = h(n.m_fWeightedOutput);
		}

		pPrev = pCurrent;
		pCurrent = pCurrent->m_pNextLayer;
	} while (pCurrent != nullptr);
}

inline void Layer::BackProp(
	const std::vector<float>& _X,
	const std::vector<float>& _Y,
	const float _fLearningRate,
	const uint32_t _uTrainingSetSize,
	const ECostFunction _kCost)
{
	// assuming this is the input layer
	if (m_pPrevLayer != nullptr)
		return;

	// forward pass
	FeedForward(_X);

	Layer* pCurrent = GetOutputLayer();
	const FloatFloatFunc& gradC(g_CostPrime[_kCost]);
	const FloatFunc& aPrime(g_ActivationPrime[m_kActivation]);

	size_t uOutputSize(pCurrent->m_Neurons.size());
	if (uOutputSize != _Y.size())
		return;

	// error d_ij is reset to 0 in forward pass
	// initialize errors Layer L 
	for (size_t i = 0; i < uOutputSize; ++i)
	{
		Neuron& n(pCurrent->m_Neurons[i]);
		// d_ij = (a_ij-y_j) * sigmoid'(z_ij)
		n.m_fError = gradC(n.m_fActivation, _Y[i]) * aPrime(n.m_fActivation);
	}

	Layer* pNext = pCurrent; // Layer l+1
	pCurrent = pCurrent->m_pPrevLayer;

	do
	{
		for (Neuron& nL1 : pNext->m_Neurons) // neuron l+1
		{
			const size_t lNeuronCount(nL1.m_InputIndices.size());
			for (size_t i = 0; i < lNeuronCount; ++i)
			{
				// neuron l
				Neuron& nL(pCurrent->m_Neurons[nL1.m_InputIndices[i]]);

				// compute error gradient: d_l = w_l+1 * d_l+1 * sigmoid'(z_l)
				nL.m_fError += nL1.m_Weights[i] * nL1.m_fError * aPrime(nL.m_fWeightedOutput);

				// learning rate n:
				float fRate = _fLearningRate / (float)_uTrainingSetSize;
				// update weights and biases: w_l = w_l - n*gradC(w_l)
				// gradC(w) = a_l-1 * d_l
				nL1.m_Weights[i] -= fRate * nL.m_fActivation * nL1.m_fError; 
				nL1.m_fBias -= fRate * nL1.m_fError;
			}	
		}

		pNext = pCurrent;
		pCurrent = pCurrent->m_pPrevLayer;
	} while (pCurrent != nullptr);
}

inline ptrdiff_t Layer::ArgMax()
{
	ptrdiff_t dist =
	std::distance(m_Neurons.begin(),
	std::max_element(m_Neurons.begin(), m_Neurons.end(),
	//[](const Neuron& n1, const Neuron& n2) {return std::fabsf(n1.m_fError) > std::fabsf(n2.m_fError); }));
	[](const Neuron& n1, const Neuron& n2) {return n1.m_fActivation < n2.m_fActivation; }));

	return dist;
}