#pragma once
#include <fstream>
#include <intrin.h>
#include <vector>

enum EMINSTType
{
	kImages = 0x00000803,
	kLabels = 0x00000801,
	kInvalid = 0xffffffff
};

class MNISTFile
{
public:
	MNISTFile(const std::string& _sFilePath);
	~MNISTFile() { }

	const uint32_t GetCount() { return m_uCount; }
	const EMINSTType GetType() const { return m_kType; }
	const uint32_t GetRows() const { return m_uRows; }
	const uint32_t GetColumns() const { return m_uColumns; }

	void GetImage(const uint32_t _uIndex, std::vector<float>& _OutImage);
	int GetLabel(const uint32_t _uIndex);
	void GetLabelVector(const uint32_t _uIndex,std::vector<float>& _YOut);

private:
	std::vector<uint8_t> m_Data;
	EMINSTType m_kType = kInvalid;
	uint32_t m_uCount = 0;
	uint32_t m_uRows = 0;
	uint32_t m_uColumns = 0;
};

inline uint32_t GetInt(std::ifstream& str)
{
	uint32_t tmp = 0;
	str.read((char*)&tmp, sizeof(uint32_t));
	return _byteswap_ulong(tmp);
}

MNISTFile::MNISTFile(const std::string& _sFilePath)
{
	std::ifstream stream(_sFilePath.c_str(), std::ios_base::binary);
	if (stream.is_open() == false)
		return;

	uint32_t uType = GetInt(stream);
	switch (uType)
	{
	case kLabels:
		m_kType = kLabels;
		m_uCount = GetInt(stream);
		m_uRows = 1u;
		m_uColumns = 1u;

		break;
	case kImages:
		m_kType = kImages;

		m_uCount = GetInt(stream);
		m_uRows = GetInt(stream);
		m_uColumns = GetInt(stream);

		break;
	default:
		return;
	}

	size_t uSize = m_uCount*m_uRows*m_uColumns;
	m_Data.resize(uSize);

	stream.read((char*)m_Data.data(), uSize);
	stream.close();
}

inline void MNISTFile::GetImage(const uint32_t _uIndex, std::vector<float>& _OutImage)
{
	if (_uIndex < m_uCount)
	{
		const size_t size = m_uRows * m_uColumns;
		_OutImage.resize(size);

		for (size_t i = 0; i < size; ++i)
		{
			_OutImage[i] = (float)m_Data[_uIndex * size + i] / 255.f;
		}
	}		
}

inline int MNISTFile::GetLabel(const uint32_t _uIndex)
{
	if (_uIndex < m_uCount)
	{
		return m_Data[_uIndex];
	}

	return 0;
}

inline void MNISTFile::GetLabelVector(const uint32_t _uIndex, std::vector<float>& _YOut)
{
	_YOut.assign(10u, 0.f);
	_YOut[GetLabel(_uIndex)] = 1.f;
}
