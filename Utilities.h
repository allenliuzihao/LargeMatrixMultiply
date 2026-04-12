#pragma once

template <typename T>
concept FloatOrInt = std::same_as<T, float> || std::same_as<T, int>;

