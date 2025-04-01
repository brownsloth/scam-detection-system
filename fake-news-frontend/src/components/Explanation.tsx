// components/Explanation.tsx
type Props = {
  input: string;
  explanation: [string, number][];
};

export default function Explanation({ input, explanation }: Props) {
  return (
    <div className="p-4 bg-gray-800 text-white border border-gray-700 rounded">
      <p className="mb-2 italic text-gray-400">“{input}”</p>
      <ul>
        {explanation.map(([label, score], i) => (
          <li key={i}>
            <span className="font-bold">{label}</span>: {score}
          </li>
        ))}
      </ul>
    </div>
  );
}
