#pragma once
#include "Neuron.h"
#include "ReadMNIST.h"
#include <random>

class Network
{
public:
	Network(const std::vector<uint32_t>& _LayerSizes);
	~Network() {};

	Layer& GetInputLayer() { return m_Layers.front(); }
	Layer& GetOutputLayer() { return m_Layers.back(); }

private:
	std::vector<Layer> m_Layers;
};

inline Network::Network(const std::vector<uint32_t>& _LayerSizes)
{
	if (_LayerSizes.empty())
		return;

	for (const uint32_t& neurons : _LayerSizes)
	{
		m_Layers.emplace_back(neurons);
	}

	for (auto it = m_Layers.begin(); it != m_Layers.end(); ++it)
	{
		if (it != m_Layers.begin())
		{
			it->SetPrev(&*(it - 1));
		}

		if (it + 1 != m_Layers.end())
		{
			it->SetNext(&*(it + 1));
		}
	}

	std::default_random_engine gen1, gen2;
	std::normal_distribution<float> distribution(0.f, 1.f);
	// fully connected layers

	for (auto it = m_Layers.rbegin(); it != m_Layers.rend(); ++it)
	{
		for (Neuron& n : it->m_Neurons)
		{
			if (it + 1 != m_Layers.rend())
			{
				size_t uInputs = (it + 1)->m_Neurons.size();
				for (uint32_t i = 0; i < uInputs; i++)
				{
					n.m_InputIndices.push_back(i);		
					n.m_Weights.push_back(distribution(gen1));
					n.m_fBias = distribution(gen2);
				}
			}
		}
	}
}